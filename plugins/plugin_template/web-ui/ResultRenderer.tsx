import React from "react";

interface ResultRendererProps {
  result?: any;
}

const ResultRenderer: React.FC<ResultRendererProps> = ({ result }) => {
  return (
    <div>
      <h4>Plugin Results</h4>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
};

export default ResultRenderer;