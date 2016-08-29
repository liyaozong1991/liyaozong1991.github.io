---
layout: post
comments: true
categories: 技术文档
---

今天学习了下cookie和session,简单谈谈自己的理解。

### 一、Cookie

&emsp;&emsp;Cookie是在远程浏览器中存储数据并以此跟踪和识别用户的机制。从实现上来说,
Cookie是存储在客户端上的一小段数据,浏览器通过HTTP协议和服务器端进行Cookie交互。
数据的具体的存储方式根据浏览器的不同会有所不同。  
&emsp;&emsp;好吧,上面的那段话还是有点绕的,Cookie就是服务器存在客户端的一些键值对,可以用来进行身份识别,密码存储,表单自动填充等。Cookie是由浏览器负责管理的,不同的浏览器存储的Cookie不能共享。
Cookie也有自己的存活时间,到期的Cookie浏览器不会立刻删除,当再次访问的时候,浏览器会判定其失效,
然后对其进行删除。   
&emsp;&emsp;Cookie可以给我们带来方便,但是也有一定的危险性。比如,通过复制别人浏览器的cookie,
可以在cookie失效前利用相同的浏览器以他人身份登录。Cookie也会增加服务器压力,
因为cookie内容较多时也会占用很大的带宽。Cookie在本地的存储大小是有限制的,具体数值与环境有关。一种新的存储方式为localStorage,可以突破Cookie大小的限制。当然localStorage的应用不止如此,具体不再展开。

Cookie在设置的时候需要注意一下参数。  
1、Cookie名称,即key值。  
2、Cookie值,即value值。  
3、有效时间,以秒为单位。这个值很重要,决定了Cookie的存储方式。如果没有设置,
Cookie保存在浏览器内存中,随着浏览器关闭而消失。一旦设置了超时时间,
Cookie就会存储在文件中,由浏览器负责管理。  
4、有效目录,默认值为"/",即整个域名下有效。  
5、Cookie的作用域名,默认本域名下。  
6、加密参数,默认为false,如果为设置为true,只有使用https,这个Cookie才会被设置。

### 二、Session

&emsp;&emsp;Session即会话,指一种持续的双向的链接。Session和Cookie本质上并没有什么区别,
都是针对http协议局限性提出的一种保持客户端和服务器间会话链接状态的一种机制。
Session通常配合Cookie使用,如果浏览器禁用Cookie,Session的使用会受到影响。

&emsp;&emsp;Session是通过SessionID来判断客户端用户的,即session文件的文件名。
SessionID就是一个Cookie,那么Session到底是如何工作的呢?  
&emsp;&emsp;我们以逛超市存包为例,假如你去超市存包,存包箱会给你一个打印一个纸条,这个纸条就是你取包的凭据,存包柜资源有限,你当然不能一直存着不取,一般超市关门前必须取走,超市会清空存包柜,如果不按时取走,就只能联系超市工作人员了。但是,你可以第二天继续去存包。  
&emsp;&emsp;这个存包的过程就类似客户端与服务器建立session的过程。客户端第一次访问服务器,
会得到一个SessionID,这个SessionID是有期限的,如果隔了很久不再继续访问,SessionID就会失效,类似于超市关门了。每次重新访问服务器,SessionID的有效期就会更新。也有一些sessionID是存在浏览器内存的,浏览器关闭即失效。这么做可能是为了安全,也可能是属于其他的考虑。   
&emsp;&emsp;最后,如果浏览器禁用了Cookie,Session就一定不可用了吗?这取决于SessionID的实现方式,
如果通过追加url等方式实现SessionID,Session就是可用的。
