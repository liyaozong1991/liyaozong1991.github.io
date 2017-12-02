---
layout: post
comments: true
categories: 技术文档
---

&emsp;&emsp;什么是http协议？http协议即超文本传输协议（HyperText Transfer Protocol）。简单的说，http就是一个基于应用的通信规范：双方要进行通信，大家都要遵守一个规范——http协议。http协议是一个应用层协议，由请求和响应构成，是一个标准的客户端服务器模型。http承载与tcp协议之上，默认端口号80。

## 一、Http协议如何工作？
&emsp;&emsp;浏览网页是http协议的主要应用，但这并不代表http协议就只能应用于网页的浏览。只要通信双方都遵守http协议，其就有用武之地。比如腾讯QQ、迅雷等软件都使用http协议。   
&emsp;&emsp;那么http协议是如何工作的呢？   
&emsp;&emsp;首先，客户端发送一个请求给服务器，服务器在接收到这个请求后将生成一个响应返回给客户端。一次http操作称为一个事务，其工作过程可分为四步：

* 客户端与服务器建立连接。单击某个超链接，http协议的工作开始。
* 建立连接后，客户机发送一个请求给服务器。格式为：前边是统一资源标识符（URI)、中间是协议版本号，后边是MIME信息（包括请求修饰符、客户机信息和可能的内容）。
* 服务器接收请求后，给予相应的响应信息。格式为：首先是一个状态行（包括信息的协议版本号、一个成功或者错误的代码），然后是MIME信息（包括服务器信息、实体信息和可能的内容）。
* 客户端接收服务返回的信息并显示在用户的显示屏上，然后客户机与服务器断开连接。

#### 1、请求
&emsp;&emsp;在发起请求前，需要先建立连接。在http1.1协议中，request和response头中都有可能出现已购车connection头，其决定当client和serve通信是对于长链接如何处理。   
&emsp;&emsp;http1.1中，client和server默认对方支持长链接，如果client使用的是http1.1协议，但又不希望使用长链接，需要在header中指明conection的值为close；如果server方也不想支持长链接，则在response中需要明确说明connection的值为close。不论request还是response的header中包含了值为close的connection，都表明当前正在使用的tcp连接在请求处理完毕后会立刻断开，以后client再进行新的请求时必须创建新的tcp连接。   
&emsp;&emsp;http请求由三部分组成：请求行、消息报头、请求正文。   
&emsp;&emsp;请求行以一个方法符号开头，以空格分开，后面跟着请求的URI和协议的版本，格式如下：   
&emsp;&emsp; **Method Request-URI HTTP-Version CRLF**   
&emsp;&emsp;下面解释下参数：

* Method：请求方法。
* Request-URI：一个统一资源标识符。
* HTTP-Version：请求的http协议版本。
* CRLF：回车或者换行

&emsp;&emsp;请求方法又分为很多种，具体如下：

* GET：请求获取Request-URI所标识的资源。
* POST：在Request-URI所标志的资源后附加新的数据（请求参数）。
* HEAD：请求获取由Request-URI所标识的资源的响应消息报头。
* PUT：请求服务器存储一个资源，并利用Request-URI作为其标识。
* DELETE：请求服务器删除Request-URI所标识的资源。
* TRACE：请求服务器回送收到的请求信息，主要用于测试或者诊断。
* CONNECT：保留以备将来使用。
* OPTIONS：请求查询服务器的性能，或者查询与资源相关的选项或需求。

#### 2、响应
&emsp;&emsp;在接收和解释请求消息后，服务器返回一个http响应消息。http响应也有三个部分组成，分别是：状态行、消息报头、响应正文。   
&emsp;&emsp;格式如下：   
&emsp;&emsp; **HTTP-Version Status-Code Reason-Phrase CRLF**  
&emsp;&emsp;下面解释参数：

* HTTP-Version：服务器http协议版本号。
* Status-Code：返回的状态码。
* Reason-Phrase：状态码的文本描述。

&emsp;&emsp;http协议的状态码由三位数字组成，第一个数字定义了响应的类别，有五种可能的情况：

* 1xx：指示信息——请求已经接收，继续处理。
* 2xx：成功——请求已经被成功接收、理解、接受。
* 3xx：重定向——要完成请求必须进行更进一步的工作。
* 4xx：客户端错我——请求有语法错误或者请求无法实现。
* 5xx：服务器错误——服务器未能实现合法的请求。

&emsp;&emsp;列举一下最常用的状态码：

* 200 OK：客户端请求成功。
* 400 Bad Request：客户端请求有语法错误，不能被服务器理解。
* 401 Unauthorize：请求未经授权。
* 403 Forbidden：服务器接收到请求，但是服务器拒绝提供服务。
* 404 Not Found：请求资源不存在，例如输入了错误的URL。
* 500 Internal Server Error：服务器发生不可预期的错误。
* 503 Server Unavailable：服务器当前不能处理客户端请求，一段时间后可能恢复正常。

#### 3、报头
&emsp;&emsp;http消息报头包括普通报头、请求报头、响应报头、实体报头。每个报头域名组成形式如下：   
&emsp;&emsp;**名字+：+空格+值**

* 普通报头中有少数报头域用于所有的请求和响应消息，但并不用于被传输的实体，只用于传输的消息（如缓存控制、连接控制等）。
* 请求报头允许客户端向服务器端传递请求的附加信息以及客户端自身的信息（如UA头、Accept等）。
* 响应报头允许服务器传递不能放在状态行中的附加响应信息，以及关于服务器的信息和对Request-URI所标识的资源进行下一步访问的信息。
* 实体报头定义了关于实体正文和请求所标识的资源的元信息，例如有无有无实体正文。

&emsp;&emsp;比较重要的几个报头如下：

* Host：头域指定请求资源的Internet主机和端口号，必须标识请求URL的原始服务器或网关位置。
* User-Agent：简称UA，内容包含发出请求的用户信息。通常UA包含浏览者信息，主要是浏览器的名称版本和所用的操作系统。
* Accept：告诉服务器可以接受的文件格式。
* Cookie：Cookie分为两种，一种是客户端向服务器端发送的，使用Cookie报头，用来标记一些信息：另一种是服务器发送给浏览器的，报头为Set-Cookie。二者的主要区别是Cookie报头的value里可以有多个Cookie值，并且不需要显示指定domain等。而Set-Cookie报头里一条记录只能有一个Cookie的value，需要指明domain、path等。
* Cache-Control：指定请求和响应遵循的缓存机制。
* Refer：头域允许客户端指定请求URI的源资源地址，这可以允许服务器生成回退链表，可用来登录、优化缓存等。
* Content-Length：内容长度。
* Content-Range：响应的资源范围。
* Accept-Encoding：指定所能接受的编码方式。

## 二、Https如何工作？
&emsp;&emsp;要弄清https是如何工作的，只需要在http的基础上理解https多了什么就可以啦。那么什么是https呢？https是HyperText Transfer Protocol over Secure Socket Layer的简称。如下图所示，https可以简单理解为ssl层加http层，也就是相当于在http层和tcp层之间新加入了ssl安全层。我们平时的编程一般都是针对http层的，所以这个ssl层很多情况下对编程人员是无感知的。

![](http://ww1.sinaimg.cn/large/75e7ad61jw1f7dtpbx9d2j20n20gbaam.jpg)

&emsp;&emsp;那么这个ssl层是如何保障安全的呢？先得了解下对称加密和非对称加密。对称加密加密和解密使用的秘钥是相同的，而非对称加密使用的加密和解密秘钥则是不同的。对称加密速度快，但是安全性差，一旦秘钥被破解，所有的加密信息都可以被破解。非对称加密速度慢，但是安全性较高，因为每个客户端都有自己的私密，所以即使某个客户端的私密泄露，也不会影响其他客户端的数据安全。下面重点说明非对称加密过程：   
&emsp;&emsp;SSL客户端在TCP链接建立之后，发出一个ClientHello来发起握手，这个消息里面包含了自己可以实现的算法列表和其他一些需要的消息，SSL的服务器端会回应一个ServerHello，这里面确定了这次通信所需要的算法，然后发送自己的证书（里面包含了自己的身份和自己的公钥）。Client在收到这个消息后会生成一个秘密消息，用SSL服务器的公钥加密后传过去，SSL服务器端用自己的私钥解密后，会话秘密协商成功，双发可以用同一份会话密钥来通信了。

![](http://ww3.sinaimg.cn/large/75e7ad61jw1f7dx22qxhfj20ng0aadg4.jpg)

&emsp;&emsp;到这里，又不得不提下证书和证书信任链了。首先会有很多的受信任的三方机构（Certificate authority），服务厂商需要向这些机构购买证书。客户端系统会内置一些受信任的证书，称为根证书（Root CA）。这些根证书作为第一级别的证书，它们也会信任一些其他证书。我们信任某个证书，一定也会信任这个证书信任的其他证书，这样就构成了证书的信任链。

![](http://ww3.sinaimg.cn/large/75e7ad61jw1f7dx0xvl58j20n70fet93.jpg)
