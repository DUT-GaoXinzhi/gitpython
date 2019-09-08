import requests
from bs4 import BeautifulSoup
import json
import bs4
list1 = []
class Movie:
    def get_html(self, url):
        try:
            r = requests.get(url)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
        except ConnectionError:
            print("error")
        return r.text

    def check_html(self, html, m_list):
        pass

    def save(self, path, m_list):
        try:
            with open(path, "w", encoding="utf-8") as f:
                for i in m_list:
                    f.write(json.dumps(i, ensure_ascii=False))
                    f.write('\n')
            print("电影信息写入已经完成")
        except FileNotFoundError:
            print("文件打开失败,请检查文件路径")
        finally:
            f.close()

    def print_all(self, m_list):
        pass


class AqyMovie(Movie):
    # html为文件,list中存储爬取到的电影名称
    def check_html(self, html, m_list):
        try:
            soup = BeautifulSoup(html, "html.parser")
        except AttributeError:
            print("soup解析失败请检查是否传入了正确的html文件")
        # 找到html文件中的热播榜对应的标签的子标签
        for i in soup.find("div", attrs={"data-seq": "1"}).contents:
            # 如果i是标签的话,找到这个标签的class属性为。。的标签
            if isinstance(i, bs4.element.Tag):
                # target为包含电影信息的标签
                target = i.find_all("div", attrs={"class": "site-piclist_info"})
                for info in target:
                    # 字典是一个对象，如果我把它放到循环的外面在后面修改的时候
                    # 相当于我一直在修改同一个对象已经存储到列表中的对象也会更改
                    dic1 = {"电影名称": "", "评分": "", "简介": ""}
                    # 获取电影的名字
                    name = info.find("a")['title']
                    # 获取电影的评分
                    i_score = info.find("strong").string
                    f_score = info.find("span").contents[1]
                    score = eval(i_score+f_score)
                    # 获取电影的简介
                    describe = info.find("p", attrs={"class": "site-piclist_info_describe"}).string
                    dic1["电影名称"] = name
                    dic1["评分"] = score
                    dic1["简介"] = describe
                    m_list.append(dic1)
        # 返回爬取的电影列表
        return m_list
    # 输出电影的信息
    # 返回电影的个数

    def print_all(self, m_list):
        count = 1
        print("{:^50}".format("爱奇艺电影热播榜"))
        print("{0:^5}\t{1:{4}^18}\t{2:{4}^5}\t{3:^20}".format("序号", "电影名", "评分", "简介", chr(12288)))
        for i in m_list:
            print("{0:^5}\t{1:{4}^18}\t{2:{4}^5}\t{3:^20}".format(count, i['电影名称'], i['评分'], i['简介'], chr(12288)))
            count += 1
        return count


if __name__ == "__main__":
    url1 = "https://www.iqiyi.com/dianying_new/i_list_paihangbang.html"
    path1 = "/home/gaoxinzhi/python-test/movie.json"
    aqy = AqyMovie()
    html1 = aqy.get_html(url1)
    aqy.check_html(html1, list1)
    aqy.save(path1, list1)
    aqy.print_all(list1)


