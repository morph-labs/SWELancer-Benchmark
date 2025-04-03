import os
import asyncio
import uuid
import shlex
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional, Any, List
from morphcloud.api import MorphCloudClient
from nanoeval.solvers.computer_tasks.task import ComputerTask
from morph_utils import safe_decode, COLORS

# Define our function to convert task to MorphCloud configuration
def task_to_morph_config(task: ComputerTask) -> "MorphConfig":
    """Convert a ComputerTask to a MorphCloud configuration"""
    updated_environment = task.environment.copy()

    issue_id = updated_environment.get("ISSUE_ID", "")

    # Infer EVAL_VARIANT based on ISSUE_ID
    if "manager" in issue_id.lower():
        eval_variant = "swe_manager"
    else:
        eval_variant = "ic_swe"

    updated_environment["EVAL_VARIANT"] = eval_variant

    return MorphConfig(
        docker_image=task.docker_image,
        environment=updated_environment,
        resources={
            "vcpus": 4,
            "memory": 8192,  # 8GB in MB
            "disk_size": 20480  # 20GB in MB
        }
    )

async def run_mitmdump_in_container(instance, container_name):
    """Run mitmdump inside the container during initialization"""
    print(f"Running mitmdump in container {container_name} during initialization...")
    
    # Make sure container is started but don't wait for full initialization
    check_started = await instance.aexec(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
    if "Up" not in safe_decode(check_started.stdout):
        print("Waiting for container to at least start...")
        for _ in range(10):  # Try for up to 10 seconds
            await asyncio.sleep(1)
            check_again = await instance.aexec(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
            if "Up" in safe_decode(check_again.stdout):
                print("Container is now up")
                break
        else:
            print("Warning: Container did not start properly")
            return False
    
    # Run mitmdump with timeout
    mitm_cmd = f"docker exec {container_name} timeout 20s mitmdump --set confdir=~/.mitmproxy --mode transparent --showhost"
    mitm_result = await instance.aexec(mitm_cmd)
    
    # Print results
    print(f"mitmdump command completed with exit code: {mitm_result.exit_code}")
    if mitm_result.exit_code == 124:
        print("mitmdump exited due to timeout (expected behavior)")
    
    # Verify certificate creation
    cert_check = await instance.aexec(
        f"docker exec {container_name} bash -c \"test -f /root/.mitmproxy/mitmproxy-ca-cert.pem && echo 'Certificate exists' || echo 'Certificate not found'\""
    )
    cert_status = safe_decode(cert_check.stdout).strip()
    print(f"Certificate status: {cert_status}")
    
    return "Certificate exists" in cert_status

# Define our MorphConfig class
class MorphConfig:
    def __init__(self, docker_image: str, resources: Dict[str, int], environment: Dict[str, str] = None):
        self.docker_image = docker_image
        self.resources = resources
        self.client = MorphCloudClient(api_key=os.environ.get("MORPH_API_KEY"))
        self.base_snapshot_metadata = {
            "purpose": "swelancer-base",
            "status": "ready"
        }
        # New: Add specific SWE-Lancer metadata
        self.swelancer_metadata = {
            "purpose": "swelancer-built-image",
            "description": "SWELancer-Benchmark with Docker image built"
        }
        self.environment = environment or {}

    @asynccontextmanager
    async def build(self, cleanup_on_error=True) -> AsyncGenerator["MorphCloudValue", None]:
        """Build and return a MorphCloudValue that contains the necessary configuration"""
        # First check for an existing SWELancer snapshot with the image already built
        print("Looking for existing SWELancer snapshot...")
        swelancer_snapshots = await self.client.snapshots.alist(
            metadata=self.swelancer_metadata
        )
        
        # If no SWELancer snapshot, check for or create a base snapshot
        print("No existing SWELancer snapshot found, will build image...")
        snapshot_id = await self._ensure_base_snapshot()
        
        # Start an instance
        print(f"Starting instance from snapshot {snapshot_id}...")
        instance = await self.client.instances.astart(snapshot_id)
        await instance.await_until_ready()
        print(f"Instance is ready.: {instance.id}")
        
        try:
            # Check Docker status first
            with instance.ssh() as ssh:
                print("Checking Docker status")
                docker_status = ssh.run(["systemctl", "status", "docker.service", "--no-pager", "-n", "20"], timeout=10)
                
                if "active (running)" not in docker_status.stdout:
                    print("Docker not running. Starting Docker...")
                    ssh.run(["systemctl", "start", "docker.service"]).raise_on_error()
            
            # If we're using a SWELancer snapshot, we should already have the image
            if swelancer_snapshots:
                # Verify image exists
                image_check = await instance.aexec(f"docker images -q {self.docker_image}")
                if not safe_decode(image_check.stdout).strip():
                    print(f"Warning: Image {self.docker_image} not found in snapshot. Will attempt to build it.")
                    await self._build_swelancer_image(instance)
            else:
                # Otherwise, we need to build the image
                await self._build_swelancer_image(instance)
            
            # Start container with a unique name
            container_name = f"swelancer-{uuid.uuid4().hex[:8]}"
            
            # Run the container using the successful method from tests.py
            print(f"Starting container: {container_name}")
            print(f'environment: {self.environment.items()}')
            
            import shlex
            env_args = ' '.join([f'-e {shlex.quote(f"{key}={value}")}' for key, value in self.environment.items()])
            
            print(env_args)
            # make sure our ssh is a login
            # manually source bashrc
            network_mode = "host"

            # stdin_open True -i
            # tty True -t
            # detach True -d

            docker_run_cmd = f"""
                docker run -d \
                  --name {container_name} \
                  -u 0 \
                  --privileged \
                  --network={network_mode} \
                  -i -t \
                  {env_args} \
                  {self.docker_image}
                """

            # docker_run_cmd = f"docker run -d --name {container_name} {env_args} {self.docker_image}"

            print(f"Executing Docker run command: {docker_run_cmd}")
            run_result = await instance.aexec(docker_run_cmd)

            if run_result.exit_code != 0:
                error_message = (
                    f"Failed to start container '{container_name}' with default ENTRYPOINT/CMD.\n"
                    f"Docker command: {docker_run_cmd}\n" # Include the docker run command in error message
                    f"Exit Code: {run_result.exit_code}\n"
                    f"Stderr: {safe_decode(run_result.stderr)}\n"
                )
                print(error_message)
                # Debug output: Print container logs on failure to start
                logs_result = await instance.aexec(f"docker logs {container_name}")
                print(f"Container logs on startup failure:\n{safe_decode(logs_result.stdout)}\n{safe_decode(logs_result.stderr)}")
                # Debug output: Inspect ENTRYPOINT and CMD on failure
                inspect_result = await instance.aexec(f"docker inspect --format='{{{{.Config.Entrypoint}}}} {{{{ .Config.Cmd }}}}' {container_name}")
                print(f"Container ENTRYPOINT/CMD on failure: {safe_decode(inspect_result.stdout)}")
                raise RuntimeError(error_message)
            else:
                print(f"Container '{container_name}' started successfully using default ENTRYPOINT/CMD.")
                # Debug output: Inspect ENTRYPOINT and CMD on successful start
                inspect_result = await instance.aexec(f"docker inspect --format='{{{{.Config.Entrypoint}}}} {{{{ .Config.Cmd }}}}' {container_name}")
                print(f"Container ENTRYPOINT/CMD on successful start: {safe_decode(inspect_result.stdout)}")

            # Verify the container is running
            check_result = await instance.aexec(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
            # Debug output: Print full check_result for container status
            print(f"Container status check result: Exit Code: {check_result.exit_code}, Stdout: {safe_decode(check_result.stdout)}, Stderr: {safe_decode(check_result.stderr)}")
            print(f"Container status (just stdout): {safe_decode(check_result.stdout)}") # Keep existing shorter status printout for easier readability
            # Run mitmdump during initialization (before considering setup complete)
            mitm_success = await run_mitmdump_in_container(instance, container_name)
            if not mitm_success:
                print("Warning: mitmproxy inject may not have completed successfully")
            else:
                print("mitmproxy inject successful")
            
            # Verify the container is still running after mitmdump
            check_result = await instance.aexec(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
            print(f"Container status after mitmdump: {safe_decode(check_result.stdout)}")

            if "Up" not in safe_decode(check_result.stdout):
                print("ERROR: Container failed to start (based on 'docker ps' check)")
                # Debug output: Print container logs if 'docker ps' check fails
                logs_result_fail_check = await instance.aexec(f"docker logs {container_name}")
                print(f"Container logs on 'docker ps' failure:\n{safe_decode(logs_result_fail_check.stdout)}\n{safe_decode(logs_result_fail_check.stderr)}")
                raise RuntimeError("Container failed to start (based on 'docker ps' check)")

            print(f"Container {container_name} started successfully")

            # Create and yield MorphCloudValue
            yield MorphCloudValue(instance=instance, container_name=container_name)

        except Exception as e:
            # If an error occurs and cleanup_on_error is False, don't stop the instance
            print(f"Error occurred: {str(e)}")
            print(f"Instance ID: {instance.id}")
            if not cleanup_on_error:
                print(f"DEBUG MODE: Keeping instance '{instance.id}' running for debugging")
                # Create a placeholder container name for debugging
                debug_container_name = f"debug-container-{uuid.uuid4().hex[:8]}"
                # Just yield the value so it can be used for debugging
                yield MorphCloudValue(instance=instance, container_name=debug_container_name)
                # Exit without cleanup
                return
            raise  # Re-raise the exception
        finally:
            # Only clean up if cleanup_on_error is True
            if cleanup_on_error:
                try:
                    # Attempt to stop the container if it exists and was defined
                    if 'container_name' in locals():
                        await instance.aexec(f"docker stop {container_name}")
                        await instance.aexec(f"docker rm {container_name}")
                except Exception as e:
                    print(f"Container cleanup error (non-critical): {str(e)}")
                    
                # Stop the instance only if cleanup_on_error is True
                print(f"Stopping instance {instance.id}")
                await instance.astop()
            else:
                # Use container_name if it was defined, otherwise just mention the instance
                try:
                    print(f"DEBUG MODE: Instance {instance.id} left running with container {container_name}")
                except UnboundLocalError:
                    print(f"DEBUG MODE: Instance {instance.id} left running")
    
    async def _build_swelancer_image(self, instance) -> None:
        """Build the SWELancer Docker image using the known Dockerfile_x86 location"""
        print("Building SWELancer Docker image...")
        
        # Known path to Dockerfile_x86 (update this with the correct path)
        expected_dockerfile_path = "SWELancer-Benchmark/Dockerfile_x86"
        
        with instance.ssh() as ssh:
            # Clone the repo
            print("Cloning SWELancer repository...")
            ssh.run(["rm", "-rf", "SWELancer-Benchmark"]).raise_on_error()  # Remove if exists
            clone_result = ssh.run(["git", "clone", "https://github.com/openai/SWELancer-Benchmark.git"], timeout=60)
            if clone_result.exit_code != 0:
                print(f"Error cloning repository: {clone_result.stderr}")
                raise RuntimeError(f"Failed to clone repository: {clone_result.stderr}")
            
            # Get current directory for absolute paths
            home_dir = ssh.run(["pwd"]).stdout.strip()
            dockerfile_path = f"{home_dir}/{expected_dockerfile_path}"
            
            # Get the directory containing the Dockerfile
            build_dir = os.path.dirname(dockerfile_path)
            dockerfile_name = os.path.basename(dockerfile_path)
            
            # Build the Docker image
            print(f"Building Docker image using {dockerfile_path}...")
            build_cmd = f"""
            cd {build_dir} && 
            echo "==== BUILDING DOCKER IMAGE ====" &&
            echo "Working directory: $(pwd)" &&
            echo "Building from file: {dockerfile_name}" &&
            ls -la &&
            docker build -t {self.docker_image} -f {dockerfile_name} .
            """
            
            # Run Docker build with extended timeout
            print("Starting Docker build (may take several minutes)...")
            build_result = ssh.run(["bash", "-c", build_cmd], timeout=1200)  # 20 minute timeout
            
            if build_result.exit_code != 0:
                print(f"Docker build failed with exit code {build_result.exit_code}")
                print("==== BUILD ERROR OUTPUT ====")
                print(build_result.stderr)
                print("==== BUILD STANDARD OUTPUT ====")
                print(build_result.stdout)
                raise RuntimeError("Failed to build Docker image")
            
            print("Docker build succeeded!")
            
            # Test the image
            print("Testing the built image...")
            test_result = ssh.run(["docker", "run", "--rm", self.docker_image, "echo", "Hello from SWE-Lancer container!"])
            if test_result.exit_code != 0:
                print(f"Image test failed: {test_result.stderr}")
                raise RuntimeError("Docker image test failed")
                
            print(f"Test output: {test_result.stdout}")
            
            # List available images
            images_result = ssh.run(["docker", "images"])
            print(f"Docker images available:\n{images_result.stdout}")
        
        # Create a snapshot with the built Docker image
        print("Creating snapshot with built Docker image...")
        snapshot = await instance.asnapshot()
        await snapshot.aset_metadata(self.swelancer_metadata)
        print(f"Snapshot created successfully with ID: {snapshot.id} and metadata {self.swelancer_metadata}")

    async def _ensure_base_snapshot(self) -> str:
        """Find or create a base snapshot with Docker installed"""
        # Check for existing snapshot with our metadata + the specific Docker image
        snapshots = await self.client.snapshots.alist(
            metadata={
                **self.base_snapshot_metadata,
                "docker_image": self.docker_image
            }
        )
        
        # No snapshot with this image, check for a base Docker snapshot
        snapshots = await self.client.snapshots.alist(
            metadata=self.base_snapshot_metadata
        )

        print("Creating new base snapshot with Docker")
        snapshot_id = await self._create_docker_base_snapshot()

        return snapshot_id
    
    async def _create_docker_base_snapshot(self) -> str:
        """Create a base snapshot with Docker installed"""
        
        # Create a snapshot with our resource specs
        snapshot = await self.client.snapshots.acreate(
            vcpus=self.resources["vcpus"],
            memory=self.resources["memory"],
            disk_size=self.resources["disk_size"]
        )
        
        # Start an instance
        instance = await self.client.instances.astart(snapshot.id)
        await instance.await_until_ready()
        
        try:
            # Use SSH to install Docker (more reliable than aexec)
            with instance.ssh() as ssh:
                print("Installing Docker via SSH...")
                
                # Update and install prerequisites
                ssh.run(["apt-get", "update"]).raise_on_error()
                ssh.run(["apt-get", "install", "-y", "curl", "apt-transport-https", 
                        "ca-certificates", "gnupg", "lsb-release"]).raise_on_error()
                
                # Install iptables-legacy which often resolves Docker issues
                ssh.run(["apt-get", "install", "-y", "iptables"]).raise_on_error()
                
                # Switch to iptables-legacy before installing Docker
                try:
                    ssh.run(["update-alternatives", "--set", "iptables", "/usr/sbin/iptables-legacy"]).raise_on_error()
                    ssh.run(["update-alternatives", "--set", "ip6tables", "/usr/sbin/ip6tables-legacy"]).raise_on_error()
                except Exception as e:
                    print(f"Note: Could not set iptables-legacy: {str(e)}")
                
                # Remove any existing Docker installation
                ssh.run(["apt-get", "remove", "--purge", "-y", "docker", "docker-engine", 
                         "docker.io", "containerd", "runc", "docker-ce", "docker-ce-cli", "containerd.io"])
                ssh.run(["apt-get", "autoremove", "-y"])
                ssh.run(["rm", "-rf", "/var/lib/docker", "/var/run/docker.sock"])
                
                # Install Docker using the official script
                ssh.run(["curl", "-fsSL", "https://get.docker.com", "-o", "install-docker.sh"]).raise_on_error()
                ssh.run(["sh", "install-docker.sh"]).raise_on_error()
                
                # Create Docker daemon configuration directory
                ssh.run(["mkdir", "-p", "/etc/docker"]).raise_on_error()

                # Create Docker daemon config file with proven working configuration
                daemon_config = '''{
                  "ipv6": false,
                  "ip6tables": false,
                  "experimental": false,
                  "log-driver": "json-file",
                  "log-opts": {
                    "max-size": "10m",
                    "max-file": "3"
                  },
                  "storage-driver": "overlay2"
                }'''
                
                ssh.run(["bash", "-c", f"echo '{daemon_config}' > /etc/docker/daemon.json"]).raise_on_error()
                
                # Reload systemd and restart Docker
                ssh.run(["systemctl", "daemon-reload"]).raise_on_error()
                ssh.run(["systemctl", "restart", "docker.service"]).raise_on_error()
                
                # Verify Docker is working by running hello-world
                print("Testing Docker with hello-world...")
                ssh.run(["docker", "run", "--rm", "hello-world"], timeout=30).raise_on_error()
                print("Docker verification succeeded!")
            
            # Create a new snapshot
            print("Creating snapshot with Docker installed")
            docker_snapshot = await instance.asnapshot()
            await docker_snapshot.aset_metadata(self.base_snapshot_metadata)
            
            return docker_snapshot.id
        finally:
            await instance.astop()

class MorphCloudValue:
    """Container for MorphCloud instance and container information"""
    def __init__(self, instance, container_name: str):
        self.instance = instance
        self.container_name = container_name