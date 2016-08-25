---
layout: post
comments: true
categories: 技术文档
---

&emsp;&emsp;Memcached是高性能的分布式内存缓存服务器，通过缓存数据库查询结果，减少数据库访问次数，以及提高动态Web应用的速度和可扩展性。   
&emsp;&emsp;Memcached有如下特点：   
* 协议简单   
* 基于libevent的事件处理机制   
* 内置内存存储方式   
* 采用不互相通信的分布式   

&emsp;&emsp;Memcached以守护程序的方式运行于一个或多个服务器中，随时接受客户端的连接操作，客户端可以由多种语言编写。客户端与Memcached服务器建立连接后，接下来的事情就是存取对象了。保在Memcached的对象都放在内存中。Memcached本身是为缓存设计的服务器，因此没有过多考虑数据持久化问题。   

### 一、Memcached数据存储
#### 1、Memcached如何支持高并发
&emsp;&emsp;Memcached使用多路复用I/O模型。传统的阻塞I/O中，系统可能会因为某个用户连接还没做好I/O准备而一直阻塞等待，直到这个连接做好I/O准备。如果这时有其他用户连接到服务器，很可能会因为系统阻塞而得不到响应。而多路复用I/O是一种消息通知模型，用户连接做好I/O准备后，系统会通知我们这个连接可以进行I/O操作，这样就不会阻塞在某个用户连接上了。   
&emsp;&emsp;此外，Memcached使用多线程模式。可以指定开启线程数量。线程数量并不是越多越好，一般设置为CPU核数，这样效率更高。因为线程数越多，系统需要的线程调度时间就越多。而把线程数设置为CPU核数，系统需要的线程调度时间最少。   

#### 2、使用Slab分配算法保存数据
&emsp;&emsp;Memcached默认只能存储不大于1M的数据，这个和Slab算法有关，这种算法可以减小内存碎片，提高内存效率。Slab算法的原理是，把固定大小的内存块（1M）划分成n小块，Slab把每1M大小的内存块称为一个slab页，每次向系统申请一个slab页，然后再通过分割算法把这个slab页分割成若干大小的chunk块，把这些chunk块分给用户使用。   
&emsp;&emsp;默认情况下，Memcached可分为40多种slab页，每种slab页的chunk块大小都不相同。Memcached向slab层申请内存数据时，Slab层从slabclass中找到一个合适的slab页，然后分配其中一个空闲的chunk块给Memcached使用。   

#### 3、删除过期的Item
&emsp;&emsp;Memcached为每个Item设置一个过期时间，但不是过期就把Item删除，而是访问Item时如果到了有效期，才把item从内存中删除。   

#### 4、使用LRU算法淘汰数据
&emsp;&emsp;
