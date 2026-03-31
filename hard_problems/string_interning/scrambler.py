with open("hard_problems/string_interning/StringInterning.java", "r", encoding="utf-8") as f:
    all_lines = list(enumerate(f.readlines()))
    print_lines = [line for line in all_lines if "System." in line[1]]
    non_print_lines = [line for line in all_lines if line not in print_lines]

    last_print_line_index = print_lines[-1][0]

    lines_before_last_print = [line for line in non_print_lines if line[0] < last_print_line_index]
    lines_after_last_print = [line for line in non_print_lines if line[0] > last_print_line_index]

import random
random.shuffle(print_lines)

for i, line in enumerate(lines_before_last_print):
    if "public class" in line[1]:
        lines_before_last_print[i] = (line[0], line[1].replace("StringInterning", "StringInterningScrambled"))
        break

final_lines = lines_before_last_print + [(""," ")] + print_lines + [(""," ")] + lines_after_last_print
with open("hard_problems/string_interning/StringInterningScrambled.java", "w", encoding="utf-8") as f:
    f.writelines(line[1] for line in final_lines)