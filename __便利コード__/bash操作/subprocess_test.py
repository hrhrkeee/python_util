import subprocess

cmd = "python test.py -t subprocess -s test"

process = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')

output = process

print(output)