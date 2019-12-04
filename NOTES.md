1) Step 1: Formatting your data
The Open Images dataset is separated into a number of components:

the image index file
the bounding box annotations file
class descriptions
trainable classes files.

sh get_dataset.sh

2) Translating Class Definitions
The trainable_classes.txt file contains encoded labels, which is totally fine for training but can be a headache during evaluation. Lets quickly use the class_descriptions.csv file to create a translated trainable classes file.

python3 translate_class_descriptions.py

usage: translate_class_descriptions.py [-h] --trainable_classes_path TRAINABLE_CLASSES
                                       --class_description_path CLASS_DESCRIPTION
                                       --trainable_translated_path TRAINABLE_TRANSLATED_PATH

??? python3 translate_class_descriptions.py --class_description_path data/classes --trainable_classes_path train/class --trainable_translated_path train/translated


3) Formatting Metadata
Lets first format our annotations file. We can do that by translating our csv rows into JSON elements, and then create a running list of image ids.

We then run a simple deduplication script over our id list, and save it so that we can filter out images we don’t need, saving us bandwidth and disk space.

sudo pip3 install tqdm
python3 process_metadata.py

usage: process_metadata.py [-h] --annotations_input_path ANNO_PATH
                           --image_index_input_path INDEX_IN_PATH
                           --point_output_path POINT_PATH
                           --image_index_output_path INDEX_OUT_PATH
                           --trainable_classes_path TRAINABLE_PATH

4) Image Downloading
As many of you might have realized, downloading ~660k web scaled images is a monstrous task. Thankfully downloading images is partially an asynchronous task, which is something we can take advantage of by multi-threading our application.

python3 download_images.py

usage: download_images.py [-h] --images_path IMAGES_PATH
                          --images_output_directory IMAGES_OUTPUT_DIRECTORY

5) Image Verification and Dimension Reduction
Now we have a ton of images, but they are all different sizes, and some of them might be broken! Let’s go ahead and verify them, but instead of verifying and resizing in two separate commands, let’s get efficient and combine the verification and resize operations.

python3 process_images.py

usage: process_images.py [-h] --image_directory IMAGE_DIRECTORY_PATH
                         [--image_saving_directory RESIZED_DIRECTORY_PATH]
                         --datapoints_input_path DATAPOINTS_INPUT_PATH
                         --datapoints_output_path DATAPOINTS_OUTPUT_PATH

6) Defining the Label Map
Tensorflow requires a label_map protobuffer file for evaluation, this object essentially just maps a label index (which is an integer value used in training) with a label keyword. If you train without an evaluation step you can avoid this, however it will help when performing inference later.

python3 create_label_map.py

usage: create_label_map.py [-h] --trainable_classes_path TRAINABLE_CLASSES
                           --class_description_path CLASS_DESCRIPTION
                           --label_map_save_path LABEL_MAP_PATH

7) TFRecord Creation
Tensorflow records are an interesting construct. They’re used nearly universally across Tensoflow objects as a dataset storage medium, and harbour a bunch of complexity, but the documentation on using your own dataset is sparse.

python3 record_maker.py


