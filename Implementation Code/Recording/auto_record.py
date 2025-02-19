from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time


def selenium_script_start(driver, URL, start_button):
    try:
        driver.get(URL)
        driver.minimize_window()

        # Wait for the start button to be clickable
        WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, start_button)))
        start_recording = driver.find_element(By.ID, start_button)
        start_recording.click()
    except TimeoutException:
        print("Timeout waiting for the start button.")
    except NoSuchElementException as e:
        print("Element not found:", e)
