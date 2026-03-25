const TWEET_API = process.env.NEXT_PUBLIC_TWEET_API;

export const scoreTweet = async (data: {
  tweet: string;
  ticker: string;
  user_credibility: number;
}) => {
  const res = await fetch(`${TWEET_API}/api/tweet/score`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Tweet scoring failed");
  }

  return res.json();
};