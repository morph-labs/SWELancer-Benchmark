import asyncio
import uuid
import shlex
from typing import List, Dict, Optional, Any
from morph_utils import safe_decode, ExecutionResult
from morph_config import MorphCloudValue

class MorphCloudInterface:
    """Implementation of ComputerInterface for MorphCloud"""
    def __init__(self, cluster_value: MorphCloudValue):
        self.cluster_value = cluster_value
   
    async def send_shell_command(
            self,
            cmd: str,
            timeout: int | None = None,
            user: str | None = None,
            container_id: int = 0,  # Ignored in MorphCloud implementation
            environment: dict[str, str] | None = None,
            workdir: str | None = None,
        ) -> ExecutionResult:
        """
        Not recommended. But for quick testing. It uses docker exec under the hood so directory changes aren't preserved.

        Args:
            cmd (str): Command to run
            timeout (int, optional): Timeout in seconds. Defaults to None.
            user (str, optional): User to run the command as. Defaults to None (root).
            container_id (int, optional): Ignored in MorphCloud implementation.
            environment (dict, optional): Environment variables.
            workdir (str, optional): Working directory.
        
        Returns:
            ExecutionResult: Object with output and exit_code
        """
        print(f'sending shell command: {cmd}')
        # Maintain the same validation logic from the original
        if not isinstance(cmd, str):
            raise ValueError(f"cmd must be of type string, but it was type {type(cmd)}")
        
        # For timeout validation, we'll assume a similar limit
        docker_client_timeout_seconds = 600  # Default value
        if hasattr(self, 'limits') and 'docker_client_timeout_seconds' in self.limits:
            docker_client_timeout_seconds = self.limits['docker_client_timeout_seconds']
            
        if timeout is not None and timeout >= docker_client_timeout_seconds:
            raise ValueError(f"{timeout=} must be less than {docker_client_timeout_seconds=} (which you can configure)")
        
        # Build the docker exec command exactly as it would be in the original implementation
        docker_cmd_parts = ["docker", "exec"]
        
        # Add options that match the original implementation
        if user is not None:
            docker_cmd_parts.extend(["-u", user])
        
        if workdir is not None:
            docker_cmd_parts.extend(["-w", workdir])
        
        # Add environment variables
        if environment is not None:
            for key, value in environment.items():
                docker_cmd_parts.extend(["-e", f"{key}={value}"])
        
        # Add container name
        docker_cmd_parts.append(self.cluster_value.container_name)
        
        # Add the command with timeout handling exactly as in original
        # Detect if this is a user-tool call or needs interactive bash
        needs_login_shell = "user-tool" in cmd or "-i" in cmd or "ansible-playbook" in cmd
        
        if needs_login_shell:
            # For user-tool commands, ensure we use login shell with source bashrc
            if timeout is None:
                docker_cmd_parts.extend(["bash", "-l", "-c", f"source ~/.bashrc && {cmd}"])
                print(f"[SHELL-CMD] Using login shell for command: {cmd[:50]}..." if len(cmd) > 50 else cmd)
            else:
                docker_cmd_parts.extend(["timeout", f"{timeout}s", "bash", "-l", "-c", f"source ~/.bashrc && {cmd}"])
                print(f"[SHELL-CMD] Using login shell with timeout for command: {cmd[:50]}..." if len(cmd) > 50 else cmd)
        else:
            if timeout is None:
                docker_cmd_parts.extend(["sh", "-c", cmd])
            else:
                docker_cmd_parts.extend(["timeout", f"{timeout}s", "sh", "-c", cmd])
        
        # Execute the command using a single string to ensure it's passed correctly
        # We need to properly escape the components to handle spaces, quotes, etc.
        import shlex
        docker_cmd_str = " ".join(shlex.quote(part) for part in docker_cmd_parts)
        
        # Execute the command
        result = await self.cluster_value.instance.aexec(docker_cmd_str)
        
        # Process output to match exactly what the original would return
        # Ensure we handle both string and bytes correctly
        stdout: bytes
        if isinstance(result.stdout, bytes):
            stdout = result.stdout
        else:
            stdout = result.stdout.encode('utf-8') if result.stdout is not None else b''
        
        stderr: bytes
        if isinstance(result.stderr, bytes):
            stderr = result.stderr
        else:
            stderr = result.stderr.encode('utf-8') if result.stderr is not None else b''
        
        # Combine stdout and stderr as the original implementation would
        combined_output = stdout + (b"\n" if stdout and stderr else b"") + stderr
        
        # Log the output exactly as the original would
        if cmd != "cd /app/tests && zip -P 'secretpasswordhere' -r /app/tmp.zip . && mv /app/tmp.zip /app/tests.zip":
            print(f"cmd {cmd} -> ({result.exit_code}) {combined_output.decode('utf-8', errors='backslashreplace')}")
        
        # Return in the correct format for compatibility
        return ExecutionResult(
            output=combined_output,
            exit_code=result.exit_code
        )

    
    async def upload(self, file_data: bytes, destination: str) -> None:
        """Upload a file to the container"""
        # Create a temporary file on the VM
        temp_path = f"/tmp/upload_{uuid.uuid4().hex}"
        
        with self.cluster_value.instance.ssh() as ssh:
            # Write a temporary file via SFTP
            sftp = ssh._client.open_sftp()
            try:
                with sftp.open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                # Copy to container - use the approach that worked in tests
                ssh.run(f"docker cp {temp_path} {self.cluster_value.container_name}:{destination}").raise_on_error()
                
                # Clean up temporary file
                ssh.run(f"rm {temp_path}")
            finally:
                sftp.close()
    
    async def download(self, file_path: str) -> bytes:
        """Download a file from the container"""
        # Create a temporary file on the VM
        temp_path = f"/tmp/download_{uuid.uuid4().hex}"
        
        with self.cluster_value.instance.ssh() as ssh:
            # Copy from container to VM - use the approach that worked in tests
            ssh.run(f"docker cp {self.cluster_value.container_name}:{file_path} {temp_path}").raise_on_error()
            
            # Read the file content using SFTP
            sftp = ssh._client.open_sftp()
            try:
                with sftp.open(temp_path, 'rb') as f:
                    file_data = f.read()
                
                # Clean up temporary file
                ssh.run(f"rm {temp_path}")
                
                return file_data
            finally:
                sftp.close()
    
    async def check_shell_command(self, command: str) -> ExecutionResult:
        """Execute a command and raise an error if it fails"""
        result = await self.send_shell_command(command)
        assert result.exit_code == 0, (
            f"Command {command} failed with {result.exit_code=}\n\n{safe_decode(result.output)}"
        )
        return result

        
    async def disable_internet(self) -> None:
        """Disable internet access for the container"""
        print("Disabling internet...")
        try:
            with self.cluster_value.instance.ssh() as ssh:
                # Get the container network config (subnet) - using approach from tests
                inspect_cmd = f"docker inspect {self.cluster_value.container_name} --format='{{{{.NetworkSettings.Networks.bridge.Subnet}}}}'"
                subnet_result = ssh.run(["bash", "-c", inspect_cmd])
                if subnet_result.exit_code != 0:
                    print(f"Warning: Failed to get subnet: {subnet_result.stderr}")
                    return
                
                subnet = subnet_result.stdout.strip().replace("'", "")
                
                # Block internet using iptables with the structure from tests.py
                cmd_template = """
                # Container network rules
                iptables -I DOCKER-USER 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
                iptables -I DOCKER-USER 2 -s {subnet} -d {subnet} -j ACCEPT
                iptables -I DOCKER-USER 3 -s {subnet} -j REJECT
                # Host communications
                iptables -I INPUT 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
                iptables -I INPUT 2 -s {subnet} -j REJECT
                """
                
                # Create a temporary script file
                script_path = f"/tmp/disable_net_{uuid.uuid4().hex}.sh"
                ssh.run(["bash", "-c", f"cat > {script_path} << 'EOL'\n{cmd_template.format(subnet=subnet)}\nEOL\n"])
                ssh.run(["chmod", "+x", script_path])
                
                # Execute the script
                ssh.run(["bash", script_path])
                
                # Clean up the temporary file
                ssh.run(["rm", script_path])
                
                # Verify the block works by checking if we can access the internet
                test_result = ssh.run([
                    "docker", "exec", self.cluster_value.container_name, 
                    "ping", "-c", "1", "-W", "3", "8.8.8.8"
                ])
                
                if test_result.exit_code == 0:
                    print("Warning: Network block might not be effective")
                else:
                    print("Internet access disabled successfully")
                    
        except Exception as e:
            print(f"Warning: Failed to disable internet: {str(e)}")
            # Don't raise the error, as this is not critical functionality
    
    async def fetch_container_names(self) -> List[str]:
        """Fetch all container names running in the instance"""
        try:
            # Just return the container name we already have, like in tests
            return [self.cluster_value.container_name]
        except Exception as e:
            print(f"Error fetching container names: {str(e)}")
            return []
    
    async def stop(self) -> None:
        """Stop the container using the approach from tests"""
        print(f"Shutting down {self.cluster_value.container_name}")
        try:
            # Stop and remove the container
            await self.cluster_value.instance.aexec(f"docker stop {self.cluster_value.container_name}")
            await self.cluster_value.instance.aexec(f"docker rm {self.cluster_value.container_name}")
        except Exception as e:
            print(f"Error stopping container: {str(e)}")
            # Don't raise as we want cleanup to continue