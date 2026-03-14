import os

base = r'c:\Users\MUNJIKESH N\Downloads\Memory_Constrained_Image_Classification_Assignment_2\Memory_Constrained_Image_Classification_Assignment_2\dataset'
exts = {'.jpg','.jpeg','.png','.bmp','.gif','.webp'}

for split in ['train', 'val', 'Test Data']:
    split_path = os.path.join(base, split)
    if not os.path.exists(split_path):
        print(f'\n=== {split} === (NOT FOUND)')
        continue
    print(f'\n=== {split} ===')
    total = 0
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
        count = sum(1 for f in os.listdir(cls_path) if os.path.splitext(f)[1].lower() in exts)
        total += count
        print(f'  {cls:<20}: {count:>5} images')
    print(f'  {"TOTAL":<20}: {total:>5} images')
