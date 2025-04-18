
from playwright.sync_api import sync_playwright, expect
from playwright._impl._page import Page
from utils.email_handler import EmailHandler


def sign_in_new_dot(page: Page, email: str, password: str, mock_otp: bool=False):
    """
    Sign in into the new Expensify dot.
    """
    
    # Sign In
    with EmailHandler(email, password) as email_handler:
        # Clean inbox
        if not mock_otp:
            email_handler.clean_inbox()

        # Enter email
        page.get_by_test_id("username").fill("expensify31+51272@gmail.com")
        page.get_by_role("button", name="Continue").click()
        try:
            try: 
                page.locator('button[tabindex="0"]').click() 
                page.wait_for_timeout(1000)
            except Exception:
                pass
            expect(page.locator("text=What do you want to do today?")).to_be_visible()
                
            # Step 4: Select 'Track and budget expenses' in onobarding page and click Continue
            page.locator("text='Track and budget expenses'").click()
            page.get_by_role("button", name="Continue").click()
            page.wait_for_timeout(1000)

            # Step 5: Enter first name, last name and click continue
            page.locator('input[name="fname"]').fill("Milan")
            page.locator('input[name="lname"]').fill("T")
            page.get_by_role("button", name="Continue").last.click()
            page.wait_for_timeout(1000)
        except Exception:

        
            # OTP
            otp = "123456" if mock_otp else email_handler.read_otp_code()
            page.get_by_test_id("validateCode").fill(otp)

            # Wait sign in to complete
            page.get_by_text("Please enter the magic code").wait_for(state="hidden")




def test_51272():
    with sync_playwright() as p:
        # Step 1: Login user 
        browser = p.chromium.launch(proxy = {"server": "http://localhost:8080"}, headless = False, args = ["--no-sandbox", "--disable-dev-shm-usage", "--ignore-certificate-errors", "--disable-web-security", "--disable-features=IsolateOrigins,site-per-process"], slow_mo = 500, channel = "chromium")
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()
        page.goto("https://dev.new.expensify.com:8082/")


        sign_in_new_dot(page=page,email="expensify31+51272@gmail.com",password="glss akzu qghd ylad",mock_otp=True)
        page.goto("https://dev.new.expensify.com:8082/settings/security/delegate")
        page.get_by_test_id("selection-list-text-input").fill("somerandomuser+2@gmail.com")
        page.get_by_label("somerandomuser+2@gmail.com").click()
        page.get_by_label("Full").click()
        page.get_by_role("button", name="Add copilot").click()
        page.mouse.click(0, 0)
        expect (page.get_by_role("button", name="Add copilot")).not_to_be_visible()

