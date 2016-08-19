---
layout: post
comments: true
categories: 技术文档
---
## 对cookie和session的理解

今天学习了下cookie和session,简单谈谈自己的理解。

### 一、Cookie

Cookie是在远程浏览器中存储数据并以此跟踪和识别用户的机制。从实现上来说,Cookie是存储在客户端
上的一小段数据,浏览器通过HTTP协议和服务器端进行Cookie交互。数据的具体的存储方式根据浏览器
的不同会有所不同。

Cookie在设置的时候需要注意一下参数。  
1、Cookie名称,即key值。  
2、Cookie值,即value值。  
3、有效时间,以秒为单位。这个值很重要,决定了Cookie的存储方式。如果没有设置,Cookie保存在
浏览器内存中,随着浏览器关闭而消失。一旦设置了超时时间,Cookie就会存储在文件中,由浏览器负责管理。  
4、有效目录,默认值为"/",即整个域名下有效。  
5、Cookie的作用域名,默认本域名下。  
6、加密参数,默认为false,如果为设置为true,只有使用https,这个Cookie才会被设置。

### 二、Session

Session即会话,指一种持续的双向的链接。Session和Cookie本质上并没有什么区别,都是针对http协议局限性
提出的一种保持客户端和服务器间会话链接状态的一种机制。Session通常配合Cookie使用,如果浏览器
禁用Cookie,Session的使用会受到影响。

Session是通过SessionID来判断客户端用户的。