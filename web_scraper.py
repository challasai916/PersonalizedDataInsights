import trafilatura

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Parameters:
        url (str): The URL of the website to scrape
        
    Returns:
        str: The extracted text content
    
    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    # Send a request to the website
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    return text if text else "Could not extract content from the provided URL."

def get_website_metadata(url: str) -> dict:
    """
    Extract metadata from a website such as title, author, date, etc.
    
    Parameters:
        url (str): The URL of the website to scrape
        
    Returns:
        dict: Dictionary containing metadata fields
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {"error": "Could not download content from URL"}
    
    metadata = trafilatura.extract_metadata(downloaded)
    if not metadata:
        return {"error": "Could not extract metadata from content"}
    
    return {
        "title": metadata.get_title() or "",
        "author": metadata.get_author() or "",
        "date": metadata.get_date() or "",
        "hostname": metadata.get_hostname() or "",
        "categories": metadata.get_categories() or [],
        "tags": metadata.get_tags() or []
    }

def extract_with_filters(url: str, include_tables: bool = False, 
                      include_links: bool = False, include_comments: bool = False) -> str:
    """
    Extract content from a website with configurable options.
    
    Parameters:
        url (str): The URL of the website to scrape
        include_tables (bool): Whether to include tables in the extraction
        include_links (bool): Whether to include links in the extraction
        include_comments (bool): Whether to include comments in the extraction
        
    Returns:
        str: The extracted text content with requested elements
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "Could not download content from URL"
    
    text = trafilatura.extract(
        downloaded,
        include_tables=include_tables,
        include_links=include_links,
        include_comments=include_comments
    )
    
    return text if text else "Could not extract content from the provided URL."