import time
import random
import logging
from typing import List
import instaloader
from modules.rate_controller import MyRateController

def batch_post_downloads(urls: List[str]):

    loader = instaloader.Instaloader(
        sanitize_paths=True,
        fatal_status_codes=[302, 400],
        rate_controller=lambda ctx: MyRateController(ctx)
    )
    
    try: 
        loader.interactive_login("some_username")
    except (instaloader.exceptions.BadCredentialsException, instaloader.exceptions.InvalidArgumentException) as e:
        logging.error("Login failed. Please check your username and password or ensure your account is not locked. Error: {}".format(e))
        return
    
  
    total_posts = len(urls)
    logging.info("Starting download of {} posts.".format(total_posts))
    
    succedded_count = 0
    
    # Iterate through each URL and download the corresponding post.
    for i in range(total_posts):
        post_url = urls[i].strip()  # Ensure no leading/trailing whitespace
        try:
            # Remove any trailing slash and extract shortcode
            clean_url = post_url.rstrip('/')
            parts = clean_url.split('/')
            if not post_url or len(parts) < 2:
                logging.info(f"URL invalid {post_url}")
                continue

            post_shortcode = parts[-1] if parts[-1] else parts[-2]

            post = instaloader.Post.from_shortcode(loader.context, post_shortcode)
            
            #check if target directory exists 
            #instaloader already handles checking for exisiting posts and will skip downloading if it already exists. no need to double check here.
            post_dir = f"Post(F_6)-{succedded_count}"
            
            loader.download_post(post, target=post_dir)
            succedded_count += 1
            delay = random.uniform(3, 6)
            logging.info(f"Downloaded post {i+1}/{total_posts}. Waiting {delay:.2f} seconds before next request.")
            time.sleep(delay)

            # Additional rate limiting every 1000 posts
            if (i+1) % 1000 == 0:
                long_delay = random.uniform(180, 300)
                logging.info(f"Reached {i+1} posts. Taking a longer break for {long_delay:.2f} seconds.")
                time.sleep(long_delay)
        except instaloader.exceptions.QueryReturnedNotFoundException as e:
            logging.error(f"Post not found for URL {post_url}: {e}")
            continue
        except instaloader.exceptions.TooManyRequestsException as e:
            logging.error(f"Rate limit hit while processing URL {post_url}: {e}")
            time.sleep(random.uniform(300, 600))
            continue
        except Exception as e:
            logging.error(f"Exception type: {type(e).__name__} - Failed to download post at URL {post_url}: {e}")
            
                
    
    if succedded_count == 0:
        logging.warning("No posts were processed. Check the input URLs.")
        return
    failed_count = total_posts - succedded_count
    logging.info(f"Completed downloading {succedded_count} posts with {failed_count} failures. Check 'failed_urls.txt' for details.")

