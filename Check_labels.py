import os

label_dirs = [
    r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\Dataset\train\labels",
    r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\Dataset\valid\labels"
]

for label_dir in label_dirs:
    print(f"\nChecking {label_dir}")

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()

            if len(parts) > 5:
                print(
                    f"SEGMENT FOUND -> {file}, line {i+1}, values={len(parts)}"
                )