---
layout: post
comments: true
categories: 技术文档
---

&emsp;&emsp;最近想要做一个文本自动分类器，主要是想试试不用的机器学习方法在文本分类上效果如何。训练分类器需要大规模的训练样本，于是需要利用python爬虫去爬取一些网页新闻作为样本，本想偷懒去网上直接找个能有的教程学习下scrapy，利用几个小时时间就把这个任务搞定，可是没想到居然踩了好多坑，在此记录一下。顺便还要吐槽一下，为什么这么多人喜欢转载别人的文章到自己的博客下面，有什么意义吗？需要学习收藏一下不就好了吗？这就导致无论的google还是baidu搜索中文博客教程总是搜到大量重复的文章，内容除了排版一点没变，实在浪费时间，搜索英文博客就好多了。   
&emsp;&emsp;简单说下我的思路吧，就先爬取16年的网易新闻，主要爬取六个标签下的新闻，最后用于分类测试。因为新闻的时间和评论数都是非常有用的信息，虽然在基于文本内容分类的时候用不到，但是以后做其他工作的时候也许会用到，所以也一起爬了下来。我用的python版本是python2.7.12 32位的，选择32位的2.7版本没什么特别的原因，主要是各种库目前对python3的支持还不太好，本例子倒是关系不大。Scrapy版本1.2.1。Scrapy官方文档已经把一些基本的内容写的非常详细了，这里不再赘述，可以参考https://scrapy.org/。 利用Scrapy我们需要做的工作并不多，定义一些规则基本就可以实现简单的爬虫了。以下文档结构：

![](http://ww4.sinaimg.cn/thumbnail/75e7ad61jw1fa7xl91dnvj205807v74i.jpg)

&emsp;&emsp;我们重点关注的只是这个NewsSpider.py文件，这个就是定义的爬虫。下面给出文档内容：

```
# -*- coding: utf-8 -*-

import scrapy
import re
import json
import logging
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib2 import urlopen
from MySpider.items import MyspiderItem

class NewsSpider(CrawlSpider):
    name = "news"
    allowed_domains = ["news.163.com","sports.163.com","ent.163.com","money.163.com","tech.163.com","digi.163.com"]
    start_urls = ['http://www.163.com/']
    rules=(
        Rule(LinkExtractor(allow=('/16/\d{4}/\d+/*',)),
        callback="parse_news",follow=True),

        Rule(LinkExtractor(deny=('/special/*')),follow=True)
    )

    logging.basicConfig(filename='D:\\GitHub\\OriginData\\MySpider\\spider_log.txt',level=logging.DEBUG)

    def parse_news(self,response):
        item = MyspiderItem()
        item['news_type'] = self.get_news_type(response)
        item['news_title'] = self.get_news_title(response)
        item['news_content'] = self.get_news_content(response)
        item['news_time'] = self.get_news_time(response)
        item['news_comments_num'] = self.get_news_comments_num(response)
        item['news_url'] = self.get_news_url(response)
        return item;

    def get_news_type(self, response):
        strs = response.url
        return strs.split('/')[2].split('.')[0]

    def get_news_title(self, response):
        title = response.xpath("/html/head/title/text()").extract()
        if title:
            return title[0]

    def get_news_url(self, response):
        return response.url

    def get_news_content(self, response):
        news_body = response.xpath("//div[@id='endText']/p/text() | //div[@id='endText']/p/a/text()").extract()
        content = ""
        if news_body:
            for s in news_body:
                content += s.strip()
        return content

    def get_news_comments_num(self, response):
        scriptBody = response.xpath('//*[@id="post_comment_area"]/script[3]/text()').extract()
        if scriptBody:
            try:
                scriptBody = scriptBody[0]
                productKey = re.search("\"productKey\" : \"\w*\"",scriptBody)
                docId = re.search("\"docId\" : \"\w*\"",scriptBody)
                productKey = productKey.group(0)[16:-1]
                docId = docId.group(0)[11:-1]
                url = 'http://comment.news.163.com/api/v1/products/'+productKey+'/threads/'+docId
                body = urlopen(url)
                # Convert bytes to string type and string type to dict
                strs = body.read().decode('utf-8')
                json_obj = json.loads(strs)
                return json_obj['cmtVote'] # prints the string with 'source_name' key
            except Exception,e:
                logging.warning(e)
        else :
            return 0

    def get_news_time(self, response):
        s = response.xpath("//div[@class='post_time_source']/text()").extract()
        if s:
            s = s[0].strip()
            s = s[:-4]
            return s
        s = response.xpath("//div[@class='ep-time-soure cDGray']/text()").extract()
        if s:
            s = s[0].strip()
            s = s[:-4]
            return s
        s = 'get time failure'
        return s

```

&emsp;&emsp;这个代码的结构非常的清晰，我们定义了爬虫规则，定义了一些函数分别获取的新闻的：标题、正文、url、评论数、时间。对正文中夹杂的图片不做记录，对正文中的超链接保留文本，使语句通顺。其中评论数是js异步获取的，爬取时需要特殊处理一下，其他的内容都可以在网页上直接解析得到。下面重点说下如何获取评论数，解释下爬虫规则定义。

#### 爬虫规则
&emsp;&emsp;对于爬虫是如何工作的，注意以下几个字段即可：

```
allowed_domains = ["news.163.com","sports.163.com","ent.163.com","money.163.com","tech.163.com","digi.163.com"]
start_urls = ['http://www.163.com/']
rules=(
    Rule(LinkExtractor(allow=('/16/\d{4}/\d+/*',)),      #表达式1
    callback="parse_news",follow=True),

    Rule(LinkExtractor(deny=('/special/*')),follow=True)
)
```

* allowed_domains：因为我们要爬取六个类别的新闻，就选取了以上六个子域名。
* start_urls：设为网易首页即可
* rules：可以定义多个规则，我们让网页地址满足正则表达式1的地址调用回调函数parse_news记录新闻，这个表达式的意思是允许域名下16年的新闻。然后在规则2中定义大部分网页都进行迭代爬取，除了special的，这类网页我发现是特殊用途，不需要迭代。规则是不是很清晰了，剩下的工作框架去做吧。

#### 评论数获取
&emsp;&emsp;评论数获取麻烦些，不过知道方法也很简单。每个新闻网页，查看源代码的话，都会在某个script标签下发现两个字段：ProductKey，DocId。通过这两个字段我们可以构造一个url，访问url，得到的内容是一个json格式的网页，我们把返回内容按json格式解析，得到cmtVote对应的值，就是评论数啦。代码上面给出了，我先用xpath得到了script标签，然后用正在表达式解析出两个字段对应的值，应该有更简单的方法。

#### 写入数据库
&emsp;&emsp;我把爬取到的内容写入到mysql数据库中，这里有两点需要注意，一个是python利用MySQLdb连接数据库后，是默认自动开启事务的，可以关闭事务或者记得提交。我直接关闭的事务。

```
self.db = MySQLdb.connect(host="localhost",user="root",passwd="123456",db="myspider",charset='utf8')
self.db.autocommit(1)
```

还有一个问题是中文乱码问题，为防止中文乱码，python文件最前都加入utf-8编码声明，mysql数据库也都采用utf-8编码。利用下面的命令把不是utf-8编码的字段改成utf-8编码。

![](http://ww4.sinaimg.cn/thumbnail/75e7ad61jw1fa7z5fd5akj20et043q3t.jpg)

```
show variables like 'character%'
set names utf8
```

&emsp;&emsp;这篇博客结合scrapy文档应该可以解决爬取网易新闻中的大部分问题，完整项目代码地址：
https://github.com/huiya9527/MySpider
