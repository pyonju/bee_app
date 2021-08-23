from icrawler.builtin import BingImageCrawler

root_dirs = ['./images/wasp_origin', './images/paper_wasp_origin']
keywords = ['スズメバチ', 'アシナガバチ']

for root_dir, keyword in zip(root_dirs, keywords):
    crawler = BingImageCrawler(storage={"root_dir": root_dir})
    crawler.crawl(keyword=keyword, max_num=1000)
