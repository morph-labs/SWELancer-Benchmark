import chz
import re
import os
import asyncio
import time
import uuid
import shlex
from textwrap import dedent
from typing import AsyncGenerator, Dict, Any
from typing_extensions import override
from contextlib import asynccontextmanager
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from nanoeval.solvers.computer_tasks.solver import PythonCodingEval
from nanoeval.solvers.computer_tasks.steps import FinalResult, FinalResultSuccessful, FinalResultWithException, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from morphcloud.api import MorphCloudClient

# Import from our modular components
from morph_utils import safe_decode, get_model_response, COLORS
from morph_config import MorphCloudValue, task_to_morph_config
from morph_interface import MorphCloudInterface

@chz.chz
class SimpleAgentSolver(PythonCodingSolver):
    name: str = "SimpleAgentSolver"
    model: str = "gpt-4o"

    def shortname(self) -> str:
        return "simple-solver"
    
    @asynccontextmanager
    async def _start_computer(self, task: ComputerTask) -> AsyncGenerator[ComputerInterface, None]:
        # Use our new MorphCloud-based implementation
        
        # print(f'TASK: {task}')
        morph_config = task_to_morph_config(task)
        
        # Set debugging flag
        debug_mode = os.environ.get("DEBUG_MODE", "").lower() == "true"  # Use environment variable, defaults to False
        
        async with morph_config.build(cleanup_on_error=not debug_mode) as morph_value:
            yield MorphCloudInterface(cluster_value=morph_value)
    
    @asynccontextmanager
    async def _with_cached_setup(self, task: ComputerTask) -> AsyncGenerator[ComputerInterface, None]:
        """
        Higher-order function that either uses a cached snapshot or runs normal setup.
        Returns an async generator that yields a ready-to-use computer interface.
        """
        issue_id = task.environment.get("ISSUE_ID", "")
        force_setup = os.environ.get("FORCE_SETUP", "")
        
        if issue_id and not force_setup:
            # Check for existing snapshot
            client = MorphCloudClient(api_key=os.environ.get("MORPH_API_KEY"))
            snapshots = await client.snapshots.alist(
                metadata={
                    "purpose": "swelancer-task-ready",
                    "issue_id": issue_id,
                    "docker_image": task.docker_image
                }
            )
            
            if snapshots:
                print(f"{COLORS['GREEN']}Morph found existing snapshot {snapshots[0].id} for issue {issue_id}{COLORS['RESET']}")
                try:
                    # Start instance from snapshot
                    instance = await client.instances.astart(snapshots[0].id)
                    await instance.await_until_ready()
                    
                    # Extract container name from metadata (with fallback)
                    container_name = snapshots[0].metadata.get("container_name", f"swelancer-{uuid.uuid4().hex[:8]}")
                    
                    # Check if container exists (running or stopped)
                    container_exists = await instance.aexec(f"docker ps -a --filter name={container_name} --format '{{{{.Names}}}}'")
                    
                    if container_exists.stdout.strip():
                        # Container exists, check if it's running
                        is_running = await instance.aexec(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
                        
                        if not is_running.stdout.strip():
                            print(f"Container {container_name} exists but is not running, starting it")
                            start_result = await instance.aexec(f"docker start {container_name}")
                            if start_result.exit_code != 0:
                                print(f"Failed to start container: {safe_decode(start_result.stderr)}")
                                raise RuntimeError(f"Failed to start container {container_name}")
                    else:
                        # Container doesn't exist, we need to create it
                        print(f"Container {container_name} not found, creating new container")
                        # Start new container with current task environment
                        env_args = ' '.join([f'-e {shlex.quote(f"{key}={value}")}' for key, value in task.environment.items()])
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
                              {task.docker_image}
                            """

                        # docker_run_cmd = f"docker run -d --name {container_name} {env_args} {task.docker_image}"
                        
                        print(f'Creating container with {docker_run_cmd}')
                        run_result = await instance.aexec(docker_run_cmd)
                        if run_result.exit_code != 0:
                            print(f"Failed to create container: {safe_decode(run_result.stderr)}")
                            raise RuntimeError(f"Failed to create container with cmd: {docker_run_cmd}")
                    
                    # We now have a running container
                    computer = MorphCloudInterface(MorphCloudValue(instance, container_name))
                    
                    try:
                        # Yield the ready computer
                        yield computer
                    finally:
                        # Consistent cleanup
                        debug_mode = os.environ.get("DEBUG_MODE", "")
                        if not debug_mode:
                            try:
                                # Stop the container first
                                await instance.aexec(f"docker stop {container_name}")
                                print(f"Stopped container {container_name}")
                            except Exception as e:
                                print(f"Warning: Failed to stop container: {e}")
                            
                            print(f"Stopping instance {instance.id}")
                            await instance.astop()
                        else:
                            print(f"DEBUG MODE: Leaving instance {instance.id} and container {container_name} running")
                    
                    # Exit the generator after cleanup
                    return
                except Exception as e:
                    print(f"Error using cached snapshot: {e}. Falling back to normal setup.")
                    # Fall through to normal setup
        
        # If we get here, we need to do the normal setup
        print("No issue_id snapshot found, proceeding with normal setup")
        async with self._start_computer(task) as computer:
            # Run diagnostic checks on container before task setup
            try:
                print("--- PERFORMING PRE-SETUP DIAGNOSTICS ---")
                # Check container logs to see if run.sh executed
                container_logs = await computer.send_shell_command("docker logs $(docker ps -q)")
                print(f"Container startup logs (excerpts): {safe_decode(container_logs.output)[:500]}...")
                
                # Check if run.sh created setup_done.txt
                setup_file = await computer.send_shell_command("cat /setup_done.txt || echo 'Setup file not found'")
                print(f"Setup done file check: {safe_decode(setup_file.output)}")
                
                # Check bashrc content
                bashrc = await computer.send_shell_command("cat ~/.bashrc || echo 'Bashrc not found'")
                bashrc_content = safe_decode(bashrc.output)
                bashrc_first_lines = bashrc_content.split('\n')[:10]
                print(f"Bashrc content check (first 10 lines): {bashrc_first_lines}")
                
                # Check for app directory
                app_dir = await computer.send_shell_command("ls -la /app || echo 'App directory not found'")
                print(f"App directory check: {safe_decode(app_dir.output)}")
                
                # Check if Dockerfile entry point / CMD is being honored
                processes = await computer.send_shell_command("ps aux | grep run.sh || echo 'run.sh process not found'")
                print(f"run.sh process check: {safe_decode(processes.output)}")
                
                print("--- END PRE-SETUP DIAGNOSTICS ---")
            except Exception as e:
                print(f"Diagnostic checks failed: {e}")
                
            # Run the task setup
            try:
                print("Running task.setup...")
                await task.setup(computer)
                print("Task setup completed successfully!")
                
                # Post-setup diagnostics
                print("--- PERFORMING POST-SETUP DIAGNOSTICS ---")
                # Check if setup_done.txt was created during setup
                setup_done_after = await computer.send_shell_command("cat /setup_done.txt || echo 'Setup file not found after task.setup'")
                print(f"Setup done file after task.setup: {safe_decode(setup_done_after.output)}")
                
                # Check if bashrc contains aliases after setup
                bashrc_alias = await computer.send_shell_command("grep alias ~/.bashrc || echo 'No aliases in bashrc after setup'")
                print(f"Bashrc aliases after setup: {safe_decode(bashrc_alias.output)}")
                
                # Check if directory structure is correct after setup
                dir_after = await computer.send_shell_command("ls -la /app/expensify || echo 'Expensify dir not found after setup'")
                print(f"Expensify directory after setup: {safe_decode(dir_after.output)[:200]}...")
                print("--- END POST-SETUP DIAGNOSTICS ---")
                
                # Create a snapshot after setup only if successful
                if isinstance(computer, MorphCloudInterface) and issue_id:
                    try:
                        print(f"Creating snapshot for issue {issue_id}")
                        snapshot = await computer.cluster_value.instance.asnapshot()
                        
                        # Consistent metadata
                        metadata = {
                            "purpose": "swelancer-task-ready",
                            "issue_id": issue_id,
                            "docker_image": task.docker_image,
                            "container_name": computer.cluster_value.container_name,
                            "created_at": str(int(time.time())),
                            "version": "1.0"  # For future compatibility
                        }
                        
                        await snapshot.aset_metadata(metadata)
                        print(f"Created snapshot {snapshot.id} for issue {issue_id}")
                    except Exception as e:
                        print(f"Warning: Failed to create snapshot: {e}")
            except Exception as e:
                print(f"Task setup failed: {e}")
                raise
            
            # Yield the computer after setup
            yield computer


    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        try:
            print(f"{COLORS['GREEN']}Running with MorphCloud!{COLORS['RESET']}")
            async with self._with_cached_setup(task) as computer:

                print(f"{COLORS['GREEN']}Agent Loop Started!{COLORS['RESET']}")
                # 2. Query the API / some agent
                messages = [] 
                for prompt_message in task.prompt:
                    messages.append({
                        "role": "user",
                        "content": str(prompt_message["content"]) # type: ignore
                    })
                messages.append({"role": "user", "content": """The repository is cloned in your CWD. You must send Python code in backticks in each response to me, and I will execute the code and send you back the result, for example:
                                                                   
```python
import os
print(os.listdir())
```

You can include other content in your message, as long as it has the Python code block. You must only submit one code block per message.
                             
The python code block is your means of interacting with the environment; write Python code to view files, edit files, and run commands. Do not leave any steps incomplete. You must implement all fixes yourself.
                                                                   
When you are done, send "DONE" (and nothing else - you don't need a python code block), and we will grade your submission.
                             
Please note that the Python code is not a Jupyter notebook; you must write a full Python script each time, including imports, and printing things if you want to see the output.
"""})
                max_turns = 30
                print(messages)

                loop_exit_reason = "Reached maximum number of turns"

                for remaining_turns in range(max_turns, 0, -1):
                    model_response = await get_model_response(messages)
                    print(model_response)

                    messages.append({"role": "assistant", "content": model_response})

                    execution_output = None

                    # Check for user-tool calls
                    if "<user-tool>" in model_response:
                        print('[USER-TOOL] User tool execution started')
                        
                        # Ensure user_tool directory exists with proper permissions
                        print('[USER-TOOL] Creating output directory')
                        mkdir_result = await computer.send_shell_command("mkdir -p /app/expensify/user_tool && chmod 777 /app/expensify/user_tool")
                        print(f'[USER-TOOL] Directory setup result: {mkdir_result.exit_code}')
                        
                        # Verify directory exists
                        dir_check = await computer.send_shell_command("ls -la /app/expensify/user_tool || echo 'Directory not found'")
                        print(f'[USER-TOOL] Directory check: {safe_decode(dir_check.output)[:100]}...')
                        
                        # Extract tests as usual
                        print('[USER-TOOL] Extracting test files')
                        await computer.check_shell_command("unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests")
                        
                        # Run ansible-playbook command directly
                        # We rely on send_shell_command to handle login shell and sourcing .bashrc
                        issue_id = task.environment.get("ISSUE_ID", "")
                        print(f'[USER-TOOL] Running ansible-playbook with ISSUE_ID={issue_id}')
                        user_tool_cmd = f"ISSUE_ID=\"{issue_id}\" ansible-playbook -i 'localhost,' --connection=local /app/tests/run_user_tool.yml"
                        user_tool_result = await computer.send_shell_command(user_tool_cmd)
                        print(f'[USER-TOOL] Ansible-playbook exit code: {user_tool_result.exit_code}')
                        
                        # Check if output was generated
                        output_check = await computer.send_shell_command("ls -la /app/expensify/user_tool/ || echo 'No output found'")
                        print(f'[USER-TOOL] Output check: {safe_decode(output_check.output)[:100]}...')
                        
                        # Standard response 
                        print('[USER-TOOL] User tool execution completed')
                        execution_output = 'User tool output is in /app/expensify/user_tool'
                        
                        # Cleanup
                        print('[USER-TOOL] Cleaning up test files')
                        clean_result = await computer.send_shell_command("rm -rf /app/tests")
                        print(f'[USER-TOOL] Cleanup result: {clean_result.exit_code}')
                    else:
                        # Extract Python code from the model response
                        python_blocks = re.findall(r"```python\s*\n(.*?)\n```", model_response, re.DOTALL)

                        if not python_blocks:
                            messages.append({
                                "role": "user",
                                "content": "Warning: No Python code blocks were found in the response. Please include Python code in your replies."
                            })
                            continue

                        code = dedent(python_blocks[0])

                        execution_output = await computer.send_shell_command(f"python -c {shlex.quote(code)}")
                        execution_output = safe_decode(execution_output.output)
                    
                    if model_response.lower() == "done":
                        print("Breaking because model is done!")
                        break

                    print(execution_output)

                    # Append the code and its output to the messages
                    messages.append({
                        "role": "user",
                        "content": f"{execution_output}\nTurns left: {remaining_turns - 1}"
                    })

                # 3. Grade and yield the final result
                print(f"Agent loop exited: {loop_exit_reason}")
                grade = await task.grade(computer)
                yield FinalResultSuccessful(grade=grade)
        except Exception as e:
            print(f"Error: {e}")
            yield FinalResultSuccessful(
                grade=Grade(score=0, grader_log=f"Grading failed with error: {str(e)}")
            )