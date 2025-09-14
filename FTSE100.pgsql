WITH Cleaned AS (
    SELECT date,
           price,
           LAG(price) OVER (ORDER BY date) AS prev_price
    FROM ftse_100
    WHERE price IS NOT NULL
),

Returns AS (
    SELECT date,
          (price - prev_price) / NULLIF(prev_price,0) AS daily_return,
          AVG(price) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma_5
    FROM Cleaned
)

SELECT date, 
       daily_return, 
       ma_5
FROM returns
WHERE daily_return IS NOT NULL
ORDER BY date;
