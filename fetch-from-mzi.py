from lxml import html
import requests


def fetch_from_url(url, file):
    page = requests.get(url)
    tree = html.fromstring(page.content)

    paragraphs = tree.xpath('//div[@class="entry-content"]/p/text()')

    with open(file, 'a+') as f:
        for para in paragraphs:
            f.write(para)
            f.write("\n")


def get_all_url(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)

    urlelements = tree.xpath('//h2[@class="entry-title"]/a')
    urltexts = []
    for elem in urlelements:
        current = elem.get("href")
        urltexts.append(current)
    return urltexts

save_file = 'zafor-sir.txt'
urls = get_all_url('https://www.ebanglalibrary.com/category/%E0%A6%AE%E0%A7%81%E0%A6%B9%E0%A6%AE%E0%A7%8D%E0%A6%AE%E0%A6%A6-%E0%A6%9C%E0%A6%BE%E0%A6%AB%E0%A6%B0-%E0%A6%87%E0%A6%95%E0%A6%AC%E0%A6%BE%E0%A6%B2/%E0%A6%95%E0%A6%BF%E0%A6%B6%E0%A7%8B%E0%A6%B0-%E0%A6%89%E0%A6%AA%E0%A6%A8%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%B8-%E0%A6%9C%E0%A6%BE%E0%A6%AB%E0%A6%B0-%E0%A6%87%E0%A6%95%E0%A6%AC%E0%A6%BE%E0%A6%B2/%E0%A6%9F%E0%A7%81%E0%A6%A8%E0%A6%9F%E0%A7%81%E0%A6%A8%E0%A6%BF-%E0%A6%93-%E0%A6%9B%E0%A7%8B%E0%A6%9F%E0%A6%BE%E0%A6%9A%E0%A7%8D%E0%A6%9A%E0%A7%81/')
print("Fetching from %d urls" % len(urls))
for url in urls:
    fetch_from_url(url, save_file)
    print("Finished %s" % url)
