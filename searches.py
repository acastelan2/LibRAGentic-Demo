from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from urllib import request
from bs4 import BeautifulSoup

class GoodReadsSearch(BaseTool):
    name: str = "GoodReads"
    description: str = """
    Used to search for user reviews on Goodreads
    using a rating system of 1 to 5 stars,
    where 1 and 2 stars are considered as negative reviews,
    3 stars is considered as neutral reviews,
    and 4 and 5 starts are considered as positive reviews.
    """

    def fetchReviews(self, search_url: str) -> str:
        response = request.urlopen(search_url).read().decode("utf-8")
        soup = BeautifulSoup(response, 'html.parser')

        content_url = ""
        for attrs in soup.find_all('a'):
            link = str(attrs.get("href"))    
            if link.startswith("/book/show"):       
                content_url = link
                break
        
        book_url = "https://www.goodreads.com"+content_url
        book_response = request.urlopen(book_url).read().decode("utf-8")
        book_soup = BeautifulSoup(book_response, 'html.parser')

        rating_count = 0
        is_review = False
        reviews = ""
        for attrs in book_soup.find_all('span'):
            if is_review:
                review = str(attrs.get("class"))
                if "Formatted" in review:
                    reviews += attrs.get_text() + "\n\n"
                    is_review = False
            else:
                rating = str(attrs.get("aria-label"))
                if rating.startswith("Rating") and int(rating[7]) > 0 and rating[8]==" ":
                    reviews += rating +"\n"
                    rating_count+=1
                    is_review = True
        return reviews

    def _run(
            self, 
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Goodreads tool."""
        base_url = "https://www.goodreads.com/search?q="
        book_title = query.replace(" ","+")
        response = self.fetchReviews((base_url+book_title))
        return response