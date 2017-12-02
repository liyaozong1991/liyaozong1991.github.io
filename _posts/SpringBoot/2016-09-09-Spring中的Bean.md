---
layout: post
comments: true
categories: Spring
---

### 一、什么是Bean
&emsp;&emsp;Bean就是一个java类，这个java类交给容器来管理，从而减少类与类之间的耦合。那么什么样的类才能声明为Bean类呢？只有你认为这个类可能会在不同的地方用到（可重用），并且需要交给容器管理，那么就可以将其声明为Bean了。但是要注意Bean类也需要遵守一定的规范。

* Java Bean类必须是一个公共类，并将其访问属性设置为public。
* Java Bean类必须有一个空的构造函数：类中必须有一个不带参数的构造函数。
* 一个JavaBean类不应该有公共的实例变量，类变量都应该声明为private的。
* 给每个属性添加getter和setter方法。

### 二、Bean的作用域
&emsp;&emsp;Scope描述的是Spring容器如何新建Bean的实例的。Spring的Scope有以下几种，通过@Scope注解来实现。

1. Singleton：一个Spring容器中只有一个Bean的实例，此为Spring默认配置，全容器共享一个实例。
2. Prototype：每次调用新建一个Bean实例。
3. Request：Web项目中，给每一个http request新建一个Bean实例。
4. Session：Web项目中，给每一个http session新建一个Bean实例。
5. GlobalSession：这个只有在portal应用中有用，给每一个global http session新建一个Bean实例。

### 三、Bean的初始化和销毁
&emsp;&emsp;在我们的实际开发中，肯带会遇到在Bean使用前或者使用后必须做一些操作，Spring对Bean 的生命周期的操作提供了支持。在使用java配置和注解配置下提供如下两种方式：

1. Java配置方式：使用@Bean的initMethod和destroyMethod。
2. 注解配置：利用JSR-250的@PostConstruct和@PreDestroy。

&emsp;&emsp;看个具体的例子（来自spring boot实战）

```
@Configuration
@ComponentScan("com.wisely.highlight_spring4.ch2.prepost")
public class PrePostConfig {

	//initMethod和destroMethod指定BeanWayService类的init和destroy方法在构造之后、Bean销毁之前执行。
	@Bean(initMethod="init",destroyMethod="destroy")
	BeanWayService beanWayService(){
		return new BeanWayService();
	}

	@Bean
	JSR250WayService jsr250WayService(){
		return new JSR250WayService();
	}

}
```

```
public class BeanWayService {
    public void init() {
        System.out.println("@Bean-init-method");
    }

    public BeanWayService() {
        super();
        System.out.println("初始化构造函数-BeanWayService");
    }

    public void destroy() {
        System.out.println("@Bean-destory-method");
    }
}
```

```
public class JSR250WayService {
	@PostConstruct //在构造函数执行之后执行
    public void init(){
        System.out.println("jsr250-init-method");
    }
    public JSR250WayService() {
        super();
        System.out.println("初始化构造函数-JSR250WayService");
    }
    @PreDestroy //在bean销毁之前执行
    public void destroy(){
        System.out.println("jsr250-destory-method");
    }

}
```

```
public class Main {

	public static void main(String[] args) {
		AnnotationConfigApplicationContext context =
                new AnnotationConfigApplicationContext(PrePostConfig.class);

		BeanWayService beanWayService = context.getBean(BeanWayService.class);
		JSR250WayService jsr250WayService = context.getBean(JSR250WayService.class);

		context.close();
	}
}

//输出
初始化构造函数-JSR250WayService
jsr250-init-method
初始化构造函数-BeanWayService
@Bean-init-method
@Bean-destory-method
jsr250-destory-method

```
&emsp;&emsp;可以看到，定义的方法全部按照正确的顺序执行。
