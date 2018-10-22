# -*- coding:utf-8 -*-
import re
import uuid
import requests
import os


class DownloadImages:
    def __init__(self, download_max, key_word):
        self.download_sum = 0
        self.download_max = download_max
        self.key_word = key_word
        self.save_path = './images/download/' + key_word

    def start_download(self):
        self.download_sum = 0
        gsm = 80
        str_gsm = str(gsm)
        pn = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        while self.download_sum < self.download_max:
            str_pn = str(self.download_sum)
            url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
                  'word=' + self.key_word + '&pn=' + str_pn + '&gsm=' + str_gsm + '&ct=&ic=0&lm=-1&width=0&height=0'
            print(url)
            result = requests.get(url)
            self.downloadImages(result.text)
        print('Download finished')


    def downloadImages(self,html):
        img_urls = re.findall('"objURL":"(.*?)",', html, re.S)
        print('Find key word: ' + self.key_word + 'Now downloading...')
        for img_url in img_urls:
            print('Downloading ' + str(self.download_sum + 1) + 'image, the address is: ' + str(img_url))
            try:
                pic = requests.get(img_url, timeout=50)
                pic_name = self.save_path + '/' + str(uuid.uuid1()) + '.' + str(img_url).split('.')[-1]
                with open(pic_name, 'wb') as f:
                    f.write(pic.content)
                self.download_sum += 1
                if self.download_sum >= self.download_max:
                    break
            except Exception as e:
                print('Download failed，%s' % e)
                continue


if __name__ == '__main__':
    key_word_max = int(input('Please enter the number of categories: '))
    key_words = []
    for sum in range(key_word_max):
        key_words.append(input('Please enter the %s key word: ' % str(sum+1)))
    max_sum = int(input('Please enter the number of images you want to download: '))
    for key_word in key_words:
        downloadImages = DownloadImages(max_sum, key_word)
        downloadImages.start_download()