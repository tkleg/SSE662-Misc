import random 

all_lines = []
with open("hard_problems/map_tricks/MapTricks.java", "r") as f:
    all_lines = f.readlines()
    first_line_to_scramble = next(i for i, line in enumerate(all_lines) if "treeMap.put(\"aa\", 1);" in line )
    last_line_to_scramble = next(i for i, line in enumerate(all_lines) if "identityMap.put(d, \"W\");" in line )
    print(f"all_lines[{first_line_to_scramble}]:", all_lines[first_line_to_scramble])
    print(f"all_lines[{last_line_to_scramble}]:", all_lines[last_line_to_scramble])
    scramble_lines = all_lines[first_line_to_scramble:last_line_to_scramble]
    random.shuffle(scramble_lines)
    all_lines[first_line_to_scramble:last_line_to_scramble] = scramble_lines
    for i, line in enumerate(all_lines):
        if( "public class" in line):
            all_lines[i] = line.replace("MapTricks", "MapTricksScrambled")
            break

with open("hard_problems/map_tricks/MapTricksScrambled.java", "w") as f:
    f.writelines(all_lines)
