from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
import random
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome options
chrome_options = Options()
# Uncomment this line to run in headless mode if needed
# chrome_options.add_argument("--headless")

# Automatically get the latest version of chromedriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the first page
base_url = 'https://app.3mtt.training/selected/fellows?page=1'

# Prepare data for CSV
data = []

# Open the first page
driver.get(base_url)

# Loop through the pages (adjust range as needed)
for page_number in range(1, 3230):  # Change to 3230 to scrape all pages
    # Wait for the specific elements in the table to load
    try:
        WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.XPATH, "//table/tbody/tr[1]/td"))
        )
    except Exception as e:
        print(f"Error loading page {page_number}: {e}")
        break

    # Scrape the fellows data
    rows = driver.find_elements(By.XPATH, "//table/tbody/tr")
    print(f"Number of rows found on page {page_number}: {len(rows)}")  # Debugging print

    for row in rows:
        try:
            name = row.find_element(By.XPATH, "./td[1]").text.strip()
            course = row.find_element(By.XPATH, "./td[2]").text.strip()
            fellow_id = row.find_element(By.XPATH, "./td[3]/span").text.strip()  # Include span for Fellow ID
            cohort = row.find_element(By.XPATH, "./td[4]").text.strip()
            state_of_residence = row.find_element(By.XPATH, "./td[5]").text.strip()
            gender = row.find_element(By.XPATH, "./td[6]").text.strip()

            # Append data to list
            data.append([name, course, fellow_id, cohort, state_of_residence, gender])
        except Exception as e:
            print(f"Error while processing row: {e}")  # Print error if one occurs

    # Click the "Next" button using JavaScript
    try:
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="root"]/section/div[3]/div/div/div/div[2]/div/div/div/button[2]'))
        )
        driver.execute_script("arguments[0].click();", next_button)

        # Wait for the new content to load after clicking "Next"
        WebDriverWait(driver, 20).until(
            EC.staleness_of(rows[0])  # Wait until the first row is no longer attached to the DOM
        )
        # Optionally wait for the new elements to be visible
        WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.XPATH, "//table/tbody/tr[1]/td"))
        )

    except Exception as e:
        print(f"Error clicking next button on page {page_number}: {e}")
        break  # Stop the loop if the next button is not found

    # Randomized wait before loading the next page
    time.sleep(random.uniform(8, 12))  # Increased random delay

# Save data to a CSV file
csv_file_path = "fellows_data_test.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Name", "Course", "Fellow ID", "Cohort", "State of Residence", "Gender"])
    # Write the data
    writer.writerows(data)

print(f"Data saved to {csv_file_path}")

# Close the browser
driver.quit()
