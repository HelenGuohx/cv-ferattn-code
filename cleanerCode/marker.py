
from sys import argv, exit

if len(argv) < 2:
    print("Usage: pass this function *.py as an argument to get every python file in the directory. It will add a line to each function that will make it crash during runtime. Delete the lines that obstruct what you do. When the code runs well, run cleaner.py with the same argument.")
    exit(2)

files = argv[1:]
badLine = '    bad.reshape(7,13)'

for file in files:
    src = open(file, mode='r')
    temp = open('deleteMe', mode='w')
    trigger = False
    indent = ''
    for line in src:
        print(line, file=temp, end='')
        if 'def ' in line:
            # We are starting a function declaration
            trigger = True
            indent = line.split('d')[0]
        if trigger and ':' in line:
            # We are now at the end of a function declaration. Add the bad line
            print(indent+badLine, file=temp)
            trigger = False

    src.close()
    temp.close()
    # Now, rewrite src using temp.
    src = open(file, mode='w')
    temp = open('deleteMe', mode='r')
    for line in temp:
        print(line, file=src, end='')
