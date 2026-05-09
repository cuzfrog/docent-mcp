use chrono::DateTime;

pub(crate) fn unix_to_rfc3339(secs: i64, nanos: u32) -> Option<String> {
    DateTime::from_timestamp(secs, nanos).map(|dt| dt.to_rfc3339())
}
