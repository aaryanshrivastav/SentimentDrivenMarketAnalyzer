const DEMO_API = process.env.NEXT_PUBLIC_DEMO_API;

export const startDemo = async (include_granger: boolean = false) => {
  const res = await fetch(`${DEMO_API}/api/demo/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ include_granger }),
  });

  if (!res.ok) throw new Error("Failed to start demo");

  return res.json();
};

export const getDemoStatus = async (jobId: string) => {
  const res = await fetch(`${DEMO_API}/api/demo/run/${jobId}`);

  if (!res.ok) throw new Error("Failed to fetch demo status");

  return res.json();
};

export const deleteDemoJob = async (jobId: string) => {
  const res = await fetch(`${DEMO_API}/api/demo/run/${jobId}`, {
    method: "DELETE",
  });

  if (!res.ok) throw new Error("Failed to delete job");

  return res.json();
};