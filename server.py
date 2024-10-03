import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
from urllib import parse

import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

reviews = pd.read_csv("data/reviews.csv").to_dict("records")


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.valid_locations = [
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona",
        ]

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_by_location(self, reviews, location):
        if location:
            if location not in self.valid_locations:
                return []  # Return empty list if location is invalid
            return [review for review in reviews if review["Location"] == location]
        return reviews

    def filter_by_date_range(self, reviews, start_date, end_date):
        filtered_reviews = []
        for review in reviews:
            review_date = datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S")
            if (start_date and review_date < start_date) or (
                end_date and review_date > end_date
            ):
                continue
            filtered_reviews.append(review)
        return filtered_reviews

    def __call__(
        self, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_string = parse.parse_qs(environ["QUERY_STRING"])
            location_query_param = query_string.get("location", [None])[0]
            start_date_query_param = query_string.get("start_date", [None])[0]
            end_date_query_param = query_string.get("end_date", [None])[0]

            start_date = (
                datetime.strptime(start_date_query_param, "%Y-%m-%d")
                if start_date_query_param
                else None
            )
            end_date = (
                datetime.strptime(end_date_query_param, "%Y-%m-%d")
                if end_date_query_param
                else None
            )

            filtered_reviews_by_location = self.filter_by_location(
                reviews, location_query_param
            )
            filtered_reviews_by_date_range = self.filter_by_date_range(
                filtered_reviews_by_location, start_date, end_date
            )

            for review in filtered_reviews_by_date_range:
                review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

            filtered_reviews_by_date_range.sort(
                key=lambda x: x["sentiment"]["compound"], reverse=True
            )

            response_body = json.dumps(filtered_reviews_by_date_range, indent=2).encode(
                "utf-8"
            )
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ],
            )
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get("CONTENT_LENGTH", 0))
                request_body = (
                    environ["wsgi.input"].read(content_length).decode("utf-8")
                )
                post_params = parse_qs(request_body)
                location_post_param = post_params.get("Location", [None])[0]
                review_body_post_param = post_params.get("ReviewBody", [None])[0]

                if not location_post_param:
                    raise ValueError("Location is required.")
                if not review_body_post_param:
                    raise ValueError("ReviewBody is required.")
                if location_post_param not in self.valid_locations:
                    raise ValueError("Invalid location.")

                new_review_entry = {
                    "ReviewId": str(uuid.uuid4()),  # Generate UUID for the review
                    "Location": location_post_param,
                    "Timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),  # Generate current timestamp
                    "ReviewBody": review_body_post_param,
                    "sentiment": self.analyze_sentiment(
                        review_body_post_param
                    ),  # Add sentiment analysis
                }

                reviews.append(new_review_entry)
                response_body = json.dumps(new_review_entry).encode("utf-8")

                start_response(
                    "201 Created",
                    [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body))),
                    ],
                )
                return [response_body]

            except ValueError as ve:
                error_message = json.dumps({"error": str(ve)}).encode("utf-8")
                start_response(
                    "400 Bad Request",
                    [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(error_message))),
                    ],
                )
                return [error_message]
            except Exception as e:
                error_message = json.dumps({"error": str(e)}).encode("utf-8")
                start_response(
                    "400 Bad Request",
                    [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(error_message))),
                    ],
                )
                return [error_message]
