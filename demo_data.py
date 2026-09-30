"""Small synthetic corpus used only to make the first-run demo explorable."""

import pandas as pd


def build_demo_data() -> pd.DataFrame:
    genuine_posts = [
        (
            "Backend software engineer",
            "Join our product team to build APIs used by healthcare customers. The role includes code review, on-call rotation, and a technical interview.",
            "A public software company founded in 2012 with 180 employees and offices in Boston.",
            "Three years of Python experience and a degree or equivalent practical experience.",
        ),
        (
            "Registered nurse, evening shift",
            "Provide patient care on a 28-bed medical unit. Schedule includes four evening shifts per week and a union orientation.",
            "Regional hospital network serving the greater Milwaukee area since 1978.",
            "Active Wisconsin RN license, BLS certification, and two years of acute care experience.",
        ),
        (
            "Accounts payable specialist",
            "Process vendor invoices, reconcile statements, and support the monthly close. Interview process includes a screening call and panel interview.",
            "Family-owned industrial supplier with 240 staff and a distribution center in Columbus.",
            "Two years of accounts payable experience and familiarity with Excel.",
        ),
        (
            "High school science teacher",
            "Plan and teach biology classes, assess student work, and collaborate with the science department. Position follows the school-year calendar.",
            "Public school district serving 14 schools in northern Oregon.",
            "Oregon teaching license with a secondary science endorsement.",
        ),
        (
            "Warehouse team lead",
            "Coordinate daily receiving and picking assignments, maintain safety procedures, and help train new staff. The role is on site.",
            "National outdoor equipment retailer with a fulfillment center in Reno.",
            "Two years of warehouse experience. Forklift certification can be completed after hire.",
        ),
        (
            "Product designer",
            "Work with product managers and engineers to research user needs and improve our mobile banking app. Portfolio review is part of the interview.",
            "Credit union technology group based in Minneapolis, founded in 2006.",
            "Experience shipping mobile product designs and proficiency with prototyping tools.",
        ),
        (
            "Customer support representative",
            "Help customers troubleshoot account issues by phone and email. Paid training, posted shift schedules, and a standard background check are included.",
            "Utility cooperative serving 85,000 households in central Colorado.",
            "Clear written communication and one year of customer service experience.",
        ),
        (
            "Civil engineering intern",
            "Support transportation projects with field observations, drafting, and calculations. This is a paid, 12-week summer internship.",
            "Employee-owned engineering consultancy with municipal clients across Illinois.",
            "Currently enrolled in an accredited civil engineering program; sophomore standing or above.",
        ),
        (
            "Human resources coordinator",
            "Schedule interviews, maintain personnel records, and coordinate new-hire orientation with the recruiting team.",
            "Food manufacturer with two production sites and a regional office in Albany.",
            "One year of administrative experience; familiarity with HR information systems is useful.",
        ),
        (
            "Solar installation technician",
            "Install rooftop solar equipment as part of a two-person crew. Includes paid safety training, travel between job sites, and a company vehicle.",
            "Licensed renewable energy contractor operating in New Mexico since 2015.",
            "Valid driver's license and construction experience preferred; OSHA 10 certification required within 90 days.",
        ),
    ]
    suspicious_posts = [
        (
            "Remote payment processing agent",
            "Earn $8,000 weekly from home with only 30 minutes of work per day. No interview needed. Send your bank login to verify your account today.",
            "",
            "No experience needed. Immediate start. Provide bank details and social security number before onboarding.",
        ),
        (
            "Mystery shopper, all locations",
            "We will mail you a check today. Deposit it and return most of the funds by gift card within 24 hours. Act now, limited openings.",
            "Global brand partner. Company details provided after acceptance.",
            "Must be over 18. No interview, experience, or application documents required.",
        ),
        (
            "Personal assistant, work from anywhere",
            "Get paid $4,500 per week to receive packages at home. Reply with your home address and a copy of your ID to get started immediately.",
            "Private employer currently traveling overseas.",
            "No skills required. Must be available to receive deliveries.",
        ),
        (
            "Crypto investment account manager",
            "Guaranteed daily returns while you work from home. Pay a refundable $250 certification fee to reserve your position before midnight.",
            "Fast-growing international investment group.",
            "No finance experience needed. Payment required to unlock your training materials.",
        ),
        (
            "Data entry clerk, urgent hiring",
            "Make $2,000 per day typing simple forms. Purchase our starter software through this link and send the receipt to secure your place.",
            "Confidential company, information available after purchase.",
            "No interview. No experience necessary. Contact us on a personal messaging account.",
        ),
        (
            "Online talent scout",
            "We guarantee placement within one day. Send your social security number and passport scan by email to start your screening instantly.",
            "Recruitment partner for major companies; client names are private.",
            "Every applicant is accepted. No experience or references are needed.",
        ),
        (
            "Work from home package inspector",
            "Receive parcels and forward them to our overseas office. Your first paycheck is included after you pay a small registration charge.",
            "International logistics opportunity with flexible hours.",
            "Provide a photo of your ID and personal address. Fee is required for account activation.",
        ),
        (
            "Secret shopper, immediate start",
            "You have been selected without applying. Deposit our check, buy gift cards, and text the codes today to confirm your position.",
            "Retail evaluation company.",
            "No application or interview. Must act immediately to keep this offer.",
        ),
        (
            "Junior finance associate, remote",
            "Earn unlimited income and guaranteed $12,000 each month. Pay for a required online course to qualify for this exclusive role.",
            "Private investment opportunity, company name shared after payment.",
            "No qualifications required. Course fee due before your first day.",
        ),
        (
            "Virtual assistant, high pay",
            "Work just one hour per day for $1,500 daily. Send your bank account and social security number now; no interview is needed.",
            "A busy executive searching for a trustworthy assistant.",
            "Everyone qualifies. Immediate hire with payment information required for payroll setup.",
        ),
    ]

    rows = []
    for label, posts in ((0, genuine_posts), (1, suspicious_posts)):
        for title, description, company_profile, requirements in posts:
            rows.append(
                {
                    "title": title,
                    "description": description,
                    "company_profile": company_profile,
                    "requirements": requirements,
                    "location": "Remote" if "remote" in title.lower() else "United States",
                    "fraudulent": label,
                }
            )
    return pd.DataFrame(rows)