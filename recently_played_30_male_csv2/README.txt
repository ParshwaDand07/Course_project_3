All recently played Men's match data in CSV format
==================================================

The background
--------------

As an experiment, after being asked by a user of the site, I started
converting the YAML data provided on the site into a CSV format. That
initial version was heavily influenced by the format used by the baseball
project Retrosheet. I wasn't sure of the usefulness of my CSV format, but
nothing better was suggested so I persisted with it. Later Ashwin Raman
(https://twitter.com/AshwinRaman_) send me a detailed example of a format
he felt might work and, liking what I saw, I started to produce data in
a slightly modified version of that initial example.

This particular zip folder contains the CSV data for...
  All recently played Men's matches
...for which we have data.

How you can help
----------------

Providing feedback on the data would be the most helpful. Tell me what you
like and what you don't. Is there anything that is in the JSON data that
you'd like to be included in the CSV? Could something be included in a better
format? General views and comments help, as well as incredibly detailed
feedback. All information is of use to me at this stage. I can only improve
the data if people tell me what does works and what doesn't. I'd like to make
the data as useful as possible but I need your help to do it. Also, which of
the 2 CSV formats do you prefer, this one or the original? Ideally I'd like
to settle on a single CSV format so what should be kept from each?

Finally, any feedback as to the licence the data should be released under
would be greatly appreciated. Licensing is a strange little world and I'd
like to choose the "right" licence. My basic criteria may be that:

  * the data should be free,
  * corrections are encouraged/required to be reported to the project,
  * derivative works are allowed,
  * you can't just take data and sell it.

Feedback, pointers, comments, etc on licensing are welcome.

The format of the data
----------------------

Full documentation of this CSV format can be found at:
  https://cricsheet.org/format/csv_ashwin/
but the following is a brief summary of the details...

This format consists of 2 files per match, although you can get all of
the ball-by-ball data from just one of the files. The files for a match
are named <id>.csv (for the ball-by-ball data), and <id>_info.csv (for
the match info), where <id> is the Cricinfo id for the match. The
ball-by-ball file contains one row per delivery in the match, while the
match info file contains match information such as dates the match was
played, the outcome, and lists of the players involved in the match.

The match info file format
--------------------------

The info section contains the information on the actual match, such as
when and where it was played, any event it was part of, the type of
match etc. The fields included in the info section will each appear as
one or more rows in the data. Some of the fields are required, whereas
some are optional. If a field has multiple values, such as team, then
each value will appear on a row of it's own.

The ball-by-ball file format
----------------------------

The first row of each ball-by-ball CSV file contains the headers for the
file, with each subsequent row providing details on a single delivery.
The headers in the file are:

  * match_id
  * season
  * start_date
  * venue
  * innings
  * ball
  * batting_team
  * bowling_team
  * striker
  * non_striker
  * bowler
  * runs_off_bat
  * extras
  * wides
  * noballs
  * byes
  * legbyes
  * penalty
  * wicket_type
  * player_dismissed
  * other_wicket_type
  * other_player_dismissed

Most of the fields above should, hopefully, be self-explanatory, but some may
benefit from clarification...

"innings" contains the number of the innings within the match. If a match is
one that would normally have 2 innings, such as a T20 or ODI, then any innings
of more than 2 can be regarded as a super over.

"ball" is a combination of the over and delivery. For example, "0.3" represents
the 3rd ball of the 1st over.

"wides", "noballs", "byes", "legbyes", and "penalty" contain the total of each
particular type of extras, or are blank if not relevant to the delivery.

If a wicket occurred on a delivery then "wicket_type" will contain the method
of dismissal, while "player_dismissed" will indicate who was dismissed. There
is also the, admittedly remote, possibility that a second dismissal can be
recorded on the delivery (such as when a player retires on the same delivery
as another dismissal occurs). In this case "other_wicket_type" will record
the reason, while "other_player_dismissed" will show who was dismissed.

Matches included in this archive
--------------------------------

2023-11-10 - international - ODI - male - 1384433 - Afghanistan vs South Africa
2023-11-09 - international - ODI - male - 1384432 - Sri Lanka vs New Zealand
2023-11-08 - international - ODI - male - 1384431 - England vs Netherlands
2023-11-07 - international - ODI - male - 1384430 - Afghanistan vs Australia
2023-11-06 - club - SMA - male - 1383568 - Punjab vs Baroda
2023-11-06 - international - ODI - male - 1384429 - Sri Lanka vs Bangladesh
2023-11-06 - club - SSH - male - 1391780 - Western Australia vs New South Wales
2023-11-06 - club - SSH - male - 1391781 - South Australia vs Queensland
2023-11-05 - international - ODI - male - 1384428 - India vs South Africa
2023-11-05 - club - SSH - male - 1391779 - Tasmania vs Victoria
2023-11-05 - international - T20 - male - 1405327 - Nepal vs Oman
2023-11-04 - club - SMA - male - 1383566 - Delhi vs Punjab
2023-11-04 - club - SMA - male - 1383567 - Assam vs Baroda
2023-11-04 - international - ODI - male - 1384426 - New Zealand vs Pakistan
2023-11-04 - international - ODI - male - 1384427 - Australia vs England
2023-11-03 - international - ODI - male - 1384425 - Netherlands vs Afghanistan
2023-11-03 - international - T20 - male - 1405325 - Bahrain vs Oman
2023-11-03 - international - T20 - male - 1405326 - United Arab Emirates vs Nepal
2023-11-02 - club - SMA - male - 1383562 - Uttar Pradesh vs Punjab
2023-11-02 - club - SMA - male - 1383563 - Kerala vs Assam
2023-11-02 - international - ODI - male - 1384424 - India vs Sri Lanka
2023-11-02 - international - T20 - male - 1405321 - Oman vs Nepal
2023-11-02 - international - T20 - male - 1405322 - United Arab Emirates vs Hong Kong
2023-11-02 - international - T20 - male - 1405323 - Malaysia vs Singapore
2023-11-02 - international - T20 - male - 1405324 - Bahrain vs Kuwait
2023-11-01 - international - ODI - male - 1384423 - South Africa vs New Zealand
2023-10-31 - club - SMA - male - 1383560 - Gujarat vs Uttar Pradesh
2023-10-31 - club - SMA - male - 1383561 - Bengal vs Assam
2023-10-31 - international - ODI - male - 1384422 - Bangladesh vs Pakistan
2023-10-31 - international - T20 - male - 1405317 - Oman vs Singapore
2023-10-31 - international - T20 - male - 1405318 - Bahrain vs Hong Kong
2023-10-31 - international - T20 - male - 1405319 - Malaysia vs Nepal
2023-10-31 - international - T20 - male - 1405320 - Kuwait vs United Arab Emirates
2023-10-30 - international - ODI - male - 1384421 - Sri Lanka vs Afghanistan
2023-10-30 - international - T20 - male - 1404394 - Namibia vs Zimbabwe
2023-10-30 - international - T20 - male - 1405313 - Bahrain vs United Arab Emirates
2023-10-30 - international - T20 - male - 1405314 - Singapore vs Nepal
2023-10-30 - international - T20 - male - 1405315 - Oman vs Malaysia
2023-10-30 - international - T20 - male - 1405316 - Hong Kong vs Kuwait
2023-10-29 - international - ODI - male - 1384420 - India vs England
2023-10-29 - international - T20 - male - 1404393 - Zimbabwe vs Namibia
2023-10-28 - international - ODI - male - 1384418 - Australia vs New Zealand
2023-10-28 - international - ODI - male - 1384419 - Netherlands vs Bangladesh
2023-10-27 - international - ODI - male - 1384417 - Pakistan vs South Africa
2023-10-27 - international - T20 - male - 1403294 - Nepal vs United Arab Emirates
2023-10-27 - international - T20 - male - 1404392 - Namibia vs Zimbabwe
2023-10-26 - international - ODI - male - 1384416 - England vs Sri Lanka
2023-10-26 - club - SSH - male - 1391776 - Victoria vs New South Wales
2023-10-26 - club - SSH - male - 1391777 - Queensland vs Tasmania
2023-10-26 - club - SSH - male - 1391778 - Western Australia vs South Australia
2023-10-25 - club - SMA - male - 1383526 - Chhattisgarh vs Baroda
2023-10-25 - club - SMA - male - 1383527 - Jammu & Kashmir vs Mizoram
2023-10-25 - club - SMA - male - 1383534 - Railways vs Gujarat
2023-10-25 - club - SMA - male - 1383539 - Uttar Pradesh vs Karnataka
2023-10-25 - club - SMA - male - 1383541 - Tamil Nadu vs Madhya Pradesh
2023-10-25 - international - ODI - male - 1384415 - Australia vs Netherlands
2023-10-25 - international - T20 - male - 1403293 - Hong Kong vs United Arab Emirates
2023-10-25 - international - T20 - male - 1404391 - Namibia vs Zimbabwe
2023-10-24 - international - ODI - male - 1384414 - South Africa vs Bangladesh
2023-10-24 - international - T20 - male - 1404390 - Zimbabwe vs Namibia
2023-10-23 - club - SMA - male - 1383507 - Haryana vs Mizoram
2023-10-23 - club - SMA - male - 1383509 - Meghalaya vs Jammu & Kashmir
2023-10-23 - club - SMA - male - 1383515 - Arunachal Pradesh vs Saurashtra
2023-10-23 - club - SMA - male - 1383517 - Andhra vs Manipur
2023-10-23 - club - SMA - male - 1383522 - Delhi vs Tamil Nadu
2023-10-23 - club - SMA - male - 1383523 - Uttar Pradesh vs Tripura
2023-10-23 - international - ODI - male - 1384413 - Pakistan vs Afghanistan
2023-10-23 - international - T20 - male - 1403292 - United Arab Emirates vs Nepal
2023-10-22 - international - ODI - male - 1384412 - New Zealand vs India
2023-10-22 - international - T20 - male - 1403291 - Hong Kong vs United Arab Emirates
2023-10-21 - club - SMA - male - 1383488 - Mumbai vs Jammu & Kashmir
2023-10-21 - club - SMA - male - 1383491 - Mizoram vs Hyderabad (India)
2023-10-21 - club - SMA - male - 1383496 - Manipur vs Punjab
2023-10-21 - club - SMA - male - 1383499 - Arunachal Pradesh vs Goa
2023-10-21 - club - SMA - male - 1383503 - Karnataka vs Delhi
2023-10-21 - club - SMA - male - 1383504 - Uttar Pradesh vs Nagaland
2023-10-21 - international - ODI - male - 1384410 - Netherlands vs Sri Lanka
2023-10-21 - international - ODI - male - 1384411 - South Africa vs England
2023-10-21 - international - T20 - male - 1403290 - Nepal vs Hong Kong
2023-10-20 - international - ODI - male - 1384409 - Australia vs Pakistan
2023-10-20 - international - T20 - male - 1403305 - Chile vs Argentina
2023-10-19 - club - SMA - male - 1383472 - Chhattisgarh vs Hyderabad (India)
2023-10-19 - club - SMA - male - 1383478 - Punjab vs Railways
2023-10-19 - club - SMA - male - 1383480 - Gujarat vs Goa
2023-10-19 - club - SMA - male - 1383485 - Madhya Pradesh vs Karnataka
2023-10-19 - club - SMA - male - 1383487 - Tripura vs Tamil Nadu
2023-10-19 - international - ODI - male - 1384408 - Bangladesh vs India
2023-10-19 - international - T20 - male - 1403289 - Hong Kong vs Nepal
2023-10-19 - international - T20 - male - 1403301 - Mexico vs Argentina
2023-10-18 - international - ODI - male - 1384407 - New Zealand vs Afghanistan
2023-10-18 - international - T20 - male - 1403288 - United Arab Emirates vs Nepal
2023-10-18 - international - T20 - male - 1403297 - Chile vs Mexico
2023-10-17 - club - SMA - male - 1383452 - Meghalaya vs Mumbai
2023-10-17 - club - SMA - male - 1383453 - Haryana vs Chhattisgarh
2023-10-17 - club - SMA - male - 1383460 - Punjab vs Andhra
2023-10-17 - club - SMA - male - 1383461 - Saurashtra vs Gujarat
2023-10-17 - club - SMA - male - 1383467 - Madhya Pradesh vs Delhi
2023-10-17 - club - SMA - male - 1383469 - Nagaland vs Tripura
2023-10-17 - international - ODI - male - 1384406 - Netherlands vs South Africa
2023-10-16 - club - SMA - male - 1383434 - Haryana vs Mumbai
2023-10-16 - club - SMA - male - 1383437 - Baroda vs Jammu & Kashmir
2023-10-16 - club - SMA - male - 1383442 - Saurashtra vs Punjab
2023-10-16 - club - SMA - male - 1383445 - Manipur vs Railways
2023-10-16 - international - ODI - male - 1384405 - Sri Lanka vs Australia
2023-10-15 - international - ODI - male - 1384404 - Afghanistan vs England
2023-10-15 - club - SSH - male - 1391774 - South Australia vs New South Wales
2023-10-15 - club - SSH - male - 1391775 - Tasmania vs Western Australia
2023-10-15 - international - T20 - male - 1400055 - Gibraltar vs Luxembourg
2023-10-15 - international - T20 - male - 1400056 - Luxembourg vs Gibraltar
2023-10-15 - international - T20 - male - 1400990 - Ghana vs Sierra Leone
2023-10-15 - international - T20 - male - 1400991 - Nigeria vs Rwanda
2023-10-14 - international - ODI - male - 1384403 - Pakistan vs India
2023-10-14 - club - SSH - male - 1391773 - Victoria vs Queensland
2023-10-14 - international - T20 - male - 1400988 - Rwanda vs Nigeria
2023-10-14 - international - T20 - male - 1400989 - Ghana vs Sierra Leone
2023-10-13 - international - ODI - male - 1384402 - Bangladesh vs New Zealand
2023-10-12 - international - ODI - male - 1384401 - South Africa vs Australia
2023-10-12 - international - T20 - male - 1400986 - Ghana vs Rwanda
2023-10-12 - international - T20 - male - 1400987 - Nigeria vs Sierra Leone
2023-10-11 - international - ODI - male - 1384400 - Afghanistan vs India
2023-10-11 - international - T20 - male - 1400984 - Ghana vs Nigeria
2023-10-11 - international - T20 - male - 1400985 - Sierra Leone vs Rwanda

Further information
-------------------

You can find all of our currently available data at https://cricsheet.org/

You can contact me via the following methods:
  Email  : stephen@cricsheet.org
  Twitter: @cricsheet
