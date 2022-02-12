import scrapy

class bookSpider(scrapy.Spider):
    name="book_spider"
    
    def start_requests(self):
        url="http://books.toscrape.com/"
        yield scrapy.Request(url,callback=self.parse)
        
    def parse(self,response):
        
        for book in response.css("article.product_pod"):
            
            img_url=book.css("img.thumbnail::attr(src)").get()
            title=book.css("h3 a::attr(title)").get()
            price=book.css("div.product_price p.price_color::text").get()
        
            yield {"image_url":img_url,"book_title":title,"product_price":price}
            
        next_page_url=response.css("li.next a::attr(href)").get()
        
        if next_page_url is not None:
            next_url=response.urljoin(next_page_url)
            yield scrapy.Request(next_url,callback=self.parse)