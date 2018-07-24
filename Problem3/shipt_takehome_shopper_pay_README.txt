This data set contains data collected on amounts paid to shopper on orders placed through Shipt (adjusted, anonymized, and sampled).
Each row contains data about one order. There are variables that provide details around:
- the order and the types of requests made
- estimated times to complete portions of the order (e.g., shopping and then delivery)
- details around the location
- details around the shopper fulfilling the order
- actual times to complete portions of the order
- the actual amount paid to the shopper

Some notes on the features:
- features with "_id" in the name are identifiers for the order, metro, and shopper
- order_alcohol_subtotal: subtotal on any alcohol items within the order
- order_num_order_lines: the number of distinct products requested (e.g., 2 boxes of Cheerios counts as 1 order line)
- order_requested_qty: the number of products requested (e.g., 2 boxes of Cheerios counts as 2 here)
- order_num_special_requests: the number of free text products requested (i.e., members can pick displayed products or enter free text to request say "2 pounds of ground beef from the meat counter")
- order_estimated_driving_time_min: an estimate for the delivery time from grocery store to member's door
- order_estimated_shopping_time_min: an estimate for the shopping time within the store
- order_actual_shopping_time_min: the actual recorded shopping time within the store (this is the difference between the timestamp at which the shopper's credit card was charged and the timestamp at which the shopper indicated in the app that they began shopping the order (either when they say the enter the store, or when they add their item to the cart within the app))
- order_actual_driving_time_min: the actual recorded driving time from grocery store to the member's door
- order_has_alcohol: yes/no flag for whether the order includes alcoholic items
- shopper_current_rating: the shopper's current rating within the app
- actual_shopper_pay: the actual amount paid to the shopper for fulfilling the order
- actual_order_cost: the actual amount paid to the grocery store for the items fulfilled
- actual_all_items_fulfilled: a binary flag for whether all the requested items were fulfilled
