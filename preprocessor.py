import re
import pandas as pd


def preprocess(data: str) -> pd.DataFrame:
    """
    Parse raw WhatsApp chat export text into a structured DataFrame.
    Supports both 12-hour (AM/PM) and 24-hour timestamp formats.
    """
    # Pattern for 24-hour format: DD/MM/YYYY, HH:MM - Name: Message
    pattern_24h = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    # Pattern for 12-hour format: DD/MM/YYYY, HH:MM AM/PM - Name: Message
    pattern_12h = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][Mm]\s-\s'

    # Detect which format is used
    if re.search(pattern_12h, data):
        pattern = pattern_12h
        date_format = '%m/%d/%Y, %I:%M %p'
    else:
        pattern = pattern_24h
        date_format = '%m/%d/%Y, %H:%M'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean date strings
    df['message_date'] = df['message_date'].str.replace(' - ', '', regex=False).str.strip()

    # Try multiple date formats
    for fmt in ['%m/%d/%Y, %I:%M %p', '%m/%d/%Y, %H:%M',
                '%d/%m/%Y, %I:%M %p', '%d/%m/%Y, %H:%M',
                '%d/%m/%y, %H:%M', '%d/%m/%y, %I:%M %p']:
        try:
            df['message_date'] = pd.to_datetime(df['message_date'], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    else:
        df['message_date'] = pd.to_datetime(df['message_date'], format='mixed')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate user and message
    users = []
    messages_clean = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            messages_clean.append(entry[2].strip())
        else:
            users.append('group_notification')
            messages_clean.append(entry[0].strip())

    df['user'] = users
    df['message'] = messages_clean
    df.drop(columns=['user_message'], inplace=True)

    # Remove system notifications
    df = df[df['user'] != 'group_notification'].copy()
    df.reset_index(drop=True, inplace=True)

    # Add time features
    df['only_date']   = df['date'].dt.date
    df['year']        = df['date'].dt.year
    df['month_num']   = df['date'].dt.month
    df['month']       = df['date'].dt.month_name()
    df['day_name']    = df['date'].dt.day_name()
    df['day']         = df['date'].dt.day
    df['hour']        = df['date'].dt.hour
    df['minute']      = df['date'].dt.minute

    # Period bucket (e.g. "23-00")
    period = []
    for h in df['hour']:
        period.append(f"{h:02d}-{(h + 1) % 24:02d}")
    df['period'] = period

    return df
