input_file = "d:\\529_group-main\\helper\\pip_requirements.txt"
output_file = "d:\\529_group-main\\helper\\pip_requirement.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Skip empty lines or comments
        if not line.strip() or line.startswith("#"):
            continue
        # Replace multiple spaces with a single '='
        package = " ".join(line.split()).replace("=", "==")
        outfile.write(package + "\n")

print(f"Converted requirements saved to {output_file}") 