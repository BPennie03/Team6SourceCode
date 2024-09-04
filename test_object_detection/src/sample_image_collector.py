from icrawler.builtin import BingImageCrawler

# Types of images to collect
classes = ['death star images']

# Number of images to collect
number = 10

for c in classes:
    crawler = BingImageCrawler(
        storage={"root_dir": "test_object_detection/resources/images/samples"})
    crawler.crawl(keyword=c, filters=None, max_num=number, offset=0)
