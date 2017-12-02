---
layout: post
comments: true
categories: Spring
---

### 一、SpringMVC常用注解和基本配置
&emsp;&emsp;SpringMVC常用以下几个注解：

1. Controller：Controller注解在类上，表明这个类是SpringMVC里的Controller，并将其声明为spring的一个bean，Dispatcher Servlet会自动扫描注解此注解的类，并将web请求映射到注解了RequestMapping的方法上。这里特别指出，在声明普通的bean的时候，使用Component、Service、Repository和Controller是等同的，因为Service、Repository和Controller都组合了Component元注解，但是在SpringMVC声明控制器的时候，只能使用Controller。
2. RequestMapping：这个注解是用来映射web请求（访问路径和参数）、处理类和方法的。RequestMapping可注解在类或者方法上。注解在方法上的RequestMapping路径会继承注解在类上的路径，RequestMapping支持Servlet的request和response作为参数，也支持对request和response的媒体类型进行配置。
3. ResponseBody：可以注解在返回值前或者方法上。支持将返回值放在response体内，而不是返回一个页面。我们在很多基于Ajax的程序的时候，可以以此注解放回数据而不是页面。
4. RequestBody：允许request的参数request体内，而不是直接连接在地址后面。此注解放置在参数前。
5. PathValue：用来接收路径参数，如/news/001，可接受001作为参数，此注解放置在参数前。
6. RestController：是一个注解组合，组合了Controller和ResponseBody，这就意味着当你只开发一个和页面交互数据的控制的时候，需要此注解。

### 二、SpringMVC基本配置
&emsp;&emsp;SpringMVC的定制配置需要我们的配置类继承WebMvcConfigurerAdapter类，并在此类使用EnableWebMvc注解，来开启对SpringMVC的支持，这样我们就可以重写这个类的方法，完成我们的常用配置。

1、静态资源映射   
&emsp;&emsp;程序的静态资源等需要直接访问，这时我们可以在配置里重写addResourceHandlers方法来实现。

```
@Configuration
@EnableWebMvc //开启SpringMvc支持，若无此句，重写WebMvcConfigurerAdapter方法无效
@EnableScheduling
@ComponentScan("com.wisely.highlight_springmvc4")
public class MyMvcConfig extends WebMvcConfigurerAdapter {// 继承类

	@Bean
	public InternalResourceViewResolver viewResolver() {
		InternalResourceViewResolver viewResolver = new InternalResourceViewResolver();
		viewResolver.setPrefix("/WEB-INF/classes/views/");
		viewResolver.setSuffix(".jsp");
		viewResolver.setViewClass(JstlView.class);
		return viewResolver;
	}

	@Override
	public void addResourceHandlers(ResourceHandlerRegistry registry) {
    //addResourceHandler是对外暴露的访问路径，addResourceLocations是文件放置的位置。
		registry.addResourceHandler("/assets/**").addResourceLocations(
				"classpath:/assets/");
	}

	@Bean
	// 1
	public DemoInterceptor demoInterceptor() {
		return new DemoInterceptor();
	}

	@Override
	public void addInterceptors(InterceptorRegistry registry) {// 2
		registry.addInterceptor(demoInterceptor());
	}

	@Override
	public void addViewControllers(ViewControllerRegistry registry) {
		registry.addViewController("/index").setViewName("/index");
		registry.addViewController("/toUpload").setViewName("/upload");
		registry.addViewController("/converter").setViewName("/converter");
		registry.addViewController("/sse").setViewName("/sse");
		registry.addViewController("/async").setViewName("/async");
	}

	 @Override
	 public void configurePathMatch(PathMatchConfigurer configurer) {
	 configurer.setUseSuffixPatternMatch(false);
	 }

	@Bean
	public MultipartResolver multipartResolver() {
		CommonsMultipartResolver multipartResolver = new CommonsMultipartResolver();
		multipartResolver.setMaxUploadSize(1000000);
		return multipartResolver;
	}

	@Override
    public void extendMessageConverters(List<HttpMessageConverter<?>> converters) {
        converters.add(converter());
    }

	@Bean
	public MyMessageConverter converter(){
		return new MyMessageConverter();
	}
}
```

2、拦截器配置   
&emsp;&emsp;拦截器实现对每一个请求处理前后进行相关的业务处理，类似于Servlet的Filter。可以让普通的Bean实现HandlerInterceptor接口或者继承HandlerInterceptorAdapter类来实现自定义拦截器。下面的代码通过重写WebConfigurerAdapter的addInterceptor方法来注册自定义的拦截器，作用为计算每一次请求的处理时间。

```
public class DemoInterceptor extends HandlerInterceptorAdapter {继承类来实现自定义的拦截器

	@Override
	public boolean preHandle(HttpServletRequest request, //在请求发生前执行
			HttpServletResponse response, Object handler) throws Exception {
		long startTime = System.currentTimeMillis();
		request.setAttribute("startTime", startTime);
		return true;
	}

	@Override
	public void postHandle(HttpServletRequest request, //在请求完成后执行
			HttpServletResponse response, Object handler,
			ModelAndView modelAndView) throws Exception {
		long startTime = (Long) request.getAttribute("startTime");
		request.removeAttribute("startTime");
		long endTime = System.currentTimeMillis();
		System.out.println("本次请求处理时间为:" + new Long(endTime - startTime)+"ms");
		request.setAttribute("handlingTime", endTime - startTime);
	}

}
```
