#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   gaoxinzhi
@Contact :   651599228@qq.com
@Software:   PyCharm
@File    :   download.py
@Time    :   2020/2/15 下午5:38
@Desc    :
'''

import requests
import json
from tqdm import tqdm
import os
from random import choice


class DownLoadMusic:
    # 初始化所有需要的信息
    def __init__(self, uid):
        # 请求头的伪装信息
        self.user_agent = {}
        # api接口的url地址
        self.urlapi = "http://music.lkxin.cn/api.php"
        self.uid = uid
        self.allmusic = []
        self.musicname = []
        self.nosrc = []
        self.my_headers = [
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
            "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
            'Opera/9.25 (Windows NT 5.1; U; en)',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
            'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
            "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
            "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
        ]
        self.prox = [
            {"https": "https://183.154.52.137:9999"},
            {"https": "https://180.122.151.149:22573"},
            {"https": "https://106.110.212.180:9999"},
            {"https": "https://36.248.132.245:9999"},
        ]

    # 给出网易云的uid号获取歌单列表
    def getmusiclist(self):
        # 创建请求数据
        postlist = {"types": "userlist", "uid": self.uid}
        loop = 'y'
        # 存放歌曲名称和uid号
        index = 0
        while loop == 'y':
            try:
                music_list = requests.post(url=self.urlapi, data=postlist, headers=self.user_agent)
                loop = 'n'
            except Exception as e:
                print("歌单列表请求失败")
                if input("是否重试y/n") == 'n':
                    exit(0)
        musiclist = json.loads(music_list.text)
        playlist = musiclist["playlist"]
        print("共找到歌单{}个\n".format(len(playlist)))
        print("序号".center(10) + "歌单名称".center(20) + "id".center(20))
        if not os.path.exists("./jsonfile"):
            os.mkdir("./jsonfile")
        for i in playlist:
            print("{}".format(index).center(10) + i["name"].center(20) + "{}".format(i["id"]).center(20))
            self.allmusic.append({"name": "{}".format(i["name"]), "id": "{}".format(i["id"])})
            index += 1
        with open("./jsonfile/playerlist.json", mode="w") as f:
            f.write(
                json.dumps({"playerlist": self.allmusic, "uid": self.uid}, indent=4, sort_keys=True,
                           ensure_ascii=False))

    # 获取特定歌单的所有歌曲名称和id
    # @param: 歌单的序号
    def getmusicname(self, index):
        count = 0
        postlist = {"types": "playlist", "id": "{}".format(self.allmusic[index]["id"])}
        while (True):
            try:
                music = requests.post(url=self.urlapi, data=postlist, headers=self.user_agent)
                break
            except Exception as e:
                if input("请求失败是否重试?y/n") == 'n':
                    exit(0)
        music_list = json.loads(music.text)
        print("歌单为  {}  共找到歌曲 {} 首\n".format(self.allmusic[index]["name"], len(music_list["playlist"]["tracks"])))
        print("序号".center(10) + "歌曲名称".center(20) + "id".center(20))

        for music in music_list["playlist"]["tracks"]:
            print("{}".format(count).center(10) + music["name"].center(20) + "{}".format(music["id"]).center(20))
            self.musicname.append({"id": "{}".format(music["id"]), "name": "{}".format(music["name"]), "url": ""})
            count += 1

    # 获取所有歌曲的资源所在位置
    def getmusicsource(self):
        # 请求数据
        postlist = {"types": "url", "id": "", "source": "netease"}
        if not os.path.exists("./jsonfile"):
            os.mkdir("./jsonfile")
        for music in tqdm(self.musicname):
            postlist["id"] = music["id"]
            try:
                content = requests.post(self.urlapi, data=postlist, headers=self.user_agent)
                res = json.loads(content.text)
                if (res["br"] == -1):
                    self.nosrc.append(music["name"])
                    continue
                music["url"] = res["url"]
            except Exception as e:
                print(e)
                print("歌曲 {} 资源地址获取失败".format(music["name"]))
        with open("./jsonfile/musiclist.json", mode="w") as f:
            f.write(
                json.dumps({"musiclist": self.musicname, "uid": self.uid}, indent=4, sort_keys=True,))
        for i in self.nosrc:
            print(i)
        if len(self.nosrc) != 0:
            print("因为版权原因不能下载")

    # 开始下载音乐
    def downloadmusic(self):
        if not os.path.exists("./music"):
            os.mkdir("./music")
        for music in tqdm(self.musicname):
            self.user_agent["User-Agent"] = choice(self.my_headers)
            if (len(music["url"]) == 0):
                continue
            try:
                src = requests.get(url=music["url"], headers=self.user_agent, proxies=choice(self.prox))
            except Exception as e:
                print(e)
                print("{}无法下载".format(music["name"]))
            try:
                with open("./music/{}.mp3".format(music["name"].replace("/", "").replace("\\", "")), "wb") as f:
                    f.write(src.content)
            except Exception as e:
                print(e)

    def begin(self):
        print("你的网易云uid为{}".format(self.uid))
        print("开始获取歌单waiting...")
        self.getmusiclist()
        index = input("请输入需要下载的歌单序号")
        self.getmusicname(eval(index))
        print("开始获取歌曲的资源位置waiting....")
        self.getmusicsource()
        if input("是否开始下载y/n") == "n":
            exit(0)
        self.downloadmusic()
        print("下载完成")


if __name__ == '__main__':
    uid = input("请输入您网易云音乐的uid号")
    a = DownLoadMusic(eval(uid))
    a.begin()
