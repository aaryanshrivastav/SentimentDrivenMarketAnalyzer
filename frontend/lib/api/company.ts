const COMPANY_API = process.env.NEXT_PUBLIC_COMPANY_API;

export const scoreCompany = async (data: {
  company_name: string;
  posts_per_query: number;
  fetch_comments: boolean;
}) => {
  const res = await fetch(`${COMPANY_API}/api/company/score`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Company scoring failed");
  }

  return res.json();
};