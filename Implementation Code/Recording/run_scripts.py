from auto_play import opencv_script
from selenium import webdriver



def main():
    CHROME_DRIVER_PATH = 'chromedriver_win32\chromedriver.exe'
    URL = 'http://"Add IP"/'
    START_BUTTON = 'rec_button'
    STOP_BUTTON = 'rec_stop'

    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome()

    # Play all videos with start and stop recording for each
    opencv_script(driver, URL, START_BUTTON, STOP_BUTTON)

    driver.quit()

if __name__ == "__main__":
    main()
