"""Business domain tools grouped by namespace."""
from tools.business.finance import register_finance_tools
from tools.business.hr import register_hr_tools
from tools.business.it import register_it_tools
from tools.business.marketing import register_marketing_tools
from tools.business.sales import register_sales_tools
from tools.business.web_scraper import register_web_scraper_tools

__all__ = [
    "register_finance_tools",
    "register_hr_tools",
    "register_it_tools",
    "register_marketing_tools",
    "register_sales_tools",
    "register_web_scraper_tools",
]
