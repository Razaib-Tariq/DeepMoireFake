from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time


def selenium_script_stop(driver, URL, stop_button):
    try:

        # Wait for the stop button to be clickable
        WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, stop_button)))
        stop_recording = driver.find_element(By.ID, stop_button)
        stop_recording.click()
        time.sleep(1)
    except TimeoutException:
        print("Timeout waiting for the stop button.")
    except NoSuchElementException as e:
        print("Element not found:", e)