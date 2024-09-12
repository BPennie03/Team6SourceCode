import os
from icrawler.builtin import GoogleImageCrawler

# Types of images to collect
classes = ['dog']

# Number of images to collect
number = 150

# Path to save images
save_path = "./test_object_detection/resources/images/samples/n"

for c in classes:
    downloaded_images = 0

    # Continue crawling until we get the target number of images
    while downloaded_images < number:
        crawler = GoogleImageCrawler(
            storage={"root_dir": save_path})

        # Crawl remaining images
        crawler.crawl(keyword=c, filters=None,
                      max_num=number, offset=0)

        # Update the number of downloaded images
        downloaded_images = len(os.listdir(save_path))

        print(f"Downloaded {downloaded_images}/{number} images")
