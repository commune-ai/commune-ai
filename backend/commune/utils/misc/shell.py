import subprocess
import shlex
def run_command(command:str):

    process = subprocess.run(shlex.split(command), 
                        stdout=subprocess.PIPE, 
                        universal_newlines=True)
    stdout, stderr = process.stdout, process.stderr
    return stdout, stderr