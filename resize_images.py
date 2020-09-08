from shutil import rmtree
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image
from get_args import get_args

def resize_images(old_dir, new_dir, size, clear_processed=False):
    if (clear_processed and exists(new_dir)):
        rmtree(new_dir)

    if not exists(new_dir):
        makedirs(new_dir)
    elif len(listdir(new_dir)) != 0:
        #print('f Error: possible overwrite of processed data. Use --clear-processed=True to clear {new_dir}.')
        return

    all_stuff = listdir(old_dir)
    is_image = lambda f: f[-4:] == '.png' or f[-4:] == '.jpg' or f[-5:] == '.jpeg'
    only_images = [f for f in all_stuff if (isfile(join(old_dir, f)) and is_image(f) )]

    errors = 0
    for f in only_images:
        try:
            img = Image.open(join(old_dir, f))
            img = img.resize(size, Image.ANTIALIAS)
            img.save(new_dir + f)
        except:
            # print(f'Error resizing ${f}')
            errors += 1

    #print(f'Resized {len(only_images) - errors} images to {size[0]}x{size[1]} with {errors} errors')


args = get_args()
resize_images(args.img_dir, './processed_data' + ('/' if args.processed_dir[0] != '/' else '') + args.processed_dir + ('/' if args.processed_dir[-1] != '/' else ''), (args.image_side_len, args.image_side_len), args.clear_processed)
