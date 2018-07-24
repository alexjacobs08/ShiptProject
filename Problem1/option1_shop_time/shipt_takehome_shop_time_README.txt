This data set contains data collected on orders placed through Shipt (adjusted, anonymized, and sampled). Each row contains information about one order. There are variables that provide details around:
- the order and the types of requests made
- the member's preferences
- the grocery store and location
- the shopper fulfilling the order, and
- the actual time it took to shop the order

Some notes on the features:
- features with "_id" in the name are identifiers for the order, store, metro, and shopper
- order_num_order_lines: the number of distinct products requested (e.g., 2 boxes of Cheerios counts as 1 order line)
- order_num_special_requests: the number of free text products requested (i.e., within the app, members can pick displayed products or enter free text to request say "2 pounds of ground beef from the meat counter")
- order_time_to_delivery_min: the number of minutes from the time the order was placed to the requested delivery time
- order_delivery_hour: delivery hour of day
- order_delivery_dow: delivery day of the week
- order_delivery_month: delivery month
- member_substitution_preference: label indicating what the shopper is expected to do in case the requested item isn't in stock
- cat_*: the number of products in each of the general product categories in the feature label
- store_pick_rate_min: the general pick rate (number of minutes / order line completed) at that specific store
- shopper_num_prev_shops: number of shops completed by this shopper prior to the existing shop
- shopper_num_prev_shops_at_store: shopper_num_prev_shops at this specific location
- shopper_pick_rate_min: the shopper-specific pick rate (see above for general pick-rate definition)
- shopper_yob: shopper's year of birth
- shopper_age: shopper's age
- shopper_gender: shopper's gender
- actual_shopping_duration_min: the difference between the timestamp at which the shopper's credit card was charged and the timestamp at which the shopper indicated in the app that they began shopping the order (either when they say the enter the store, or when they add their item to the cart within the app)
