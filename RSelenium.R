library(RSelenium)

# Start Selenium server
selenium_server <- rsDriver(browser = "chrome", chromever = "latest")
remDr <- selenium_server$client

# Navigate to a dynamic webpage
url <- "https://example.com"
remDr$navigate(url)

# Wait for the dynamic content to load (adjust as needed)
Sys.sleep(5)

# Extract page source
page_source <- remDr$getPageSource()[[1]]

# Parse HTML using rvest
library(rvest)
page <- read_html(page_source)

# Extract desired elements (e.g., titles or text)
titles <- page %>% html_nodes(".title-class") %>% html_text()
print(titles)

# Close the Selenium driver
remDr$close()
selenium_server$server$stop()
