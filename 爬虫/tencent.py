from bs4 import BeautifulSoup
import bs4
import aiqiyi


class TMovie(aiqiyi.Movie):
    # 解析html界面
    def check_html(self, html, list):
        soup = BeautifulSoup(html, "html.parser")
        start = soup.find("div", attrs={"class": "mod_figure mod_figure_v_default mod_figure_list_box"})
        for i in start.contents:
            dict = {"电影名称": "", "评分": "", "简介": ""}
            if isinstance(i, bs4.element.Tag):
                info = i.find('a')
                dict["电影名称"] = info["title"]
                dict["评分"] = info.find("div", attrs={"class": "figure_score"}).string
                dict["简介"] = i.find("div", attrs={"class": "figure_desc"}).string
                list.append(dict)
        return list

    def print_all(self, m_list):
        count = 1
        print("{:^50}".format("腾讯电影推荐"))
        print("{0:^5}\t{1:{4}^18}\t{2:{4}^5}\t{3:^20}".format("序号", "电影名", "评分", "简介", chr(12288)))
        for i in m_list:
            print("{0:^5}\t{1:{4}^18}\t{2:{4}^5}\t{3:^20}".format(count, i['电影名称'], i['评分'], i['简介'], chr(12288)))
            count += 1
        return count


if __name__ == "__main__":
    url1 = "https://v.qq.com/channel/movie?listpage=1&channel=movie&itype=100062"
    path1 = "/home/gaoxinzhi/python-test/movie1.json"
    tencent = TMovie()
    html1 = tencent.get_html(url1)
    list1 = []
    tencent.check_html(html1, list1)
    tencent.save(path1, list1)
    tencent.print_all(list1)
