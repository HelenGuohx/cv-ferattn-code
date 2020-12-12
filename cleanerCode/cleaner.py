
from sys import argv, exit

if len(argv) < 2:
    print("Usage: This function deletes any function with the expression 'bad.reshape(7,13)', ideally put there by marker.py.")
    exit(2)

files = argv[1:]
badLine = 'bad.reshape(7,13)'

bad = False
for file in files:
    src = open(file, mode='r')
    temp = open('deleteMe', mode='w')
    lines = []
    for line in src:
        if 'def' in line or 'class' in line:
            if not bad:
                # the prior function must be kept
                print(lines, file=temp, end='')
            # Dump the lines
            lines = []
            bad = False
        lines.append(line)
        if badLine in line:
            bad = True

    src.close()
    temp.close()
    # Now, rewrite src using temp.
    src = open(file, mode='w')
    temp = open('deleteMe', mode='r')
    for line in temp:
        print(line, file=src, end='')



