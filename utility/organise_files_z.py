import os
import shutil

# NOTE: this file is written to organise folders in a way that the files are readable by CMAL-NET algorithm (and PIM?). 
# Since every dataset has a different starting structure, modifications may be needed based on the original structure of the data

# Path to the directory containing the images
image_dir = "/home/zazza/Documents/ML/fgvc-aircraft/data/images/Test"

# Path to the directory containing the TXT file
txt_file = "/home/zazza/Documents/ML/fgvc-aircraft/data/images_variant_test.txt"

def read_txt_file(txt_file):
    variants = {}
    with open(txt_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                planecode, variant = parts
                variants[planecode] = variant
    return variants

# create folders by class
def create_variant_folders(variants, image_dir):
    for variant in set(variants.values()):
        os.makedirs(os.path.join(image_dir, variant), exist_ok=True)

def move_images(variants, image_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            planecode = os.path.splitext(filename)[0]
            if planecode in variants:
                variant = variants[planecode]
                shutil.move(os.path.join(image_dir, filename), os.path.join(image_dir, variant, filename))

def main():
    variants = read_txt_file(txt_file)
    create_variant_folders(variants, image_dir)
    move_images(variants, image_dir)

if __name__ == "__main__":
    main()
