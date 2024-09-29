export default function handler(req, res) {
    if (req.method === 'POST') {
      // Extract data from the request body
      const { name } = req.body;
  
      // Return a success message
      res.status(200).json({ message: `Hello, ${name}!` });
    } else {
      // If not a POST request, send a method not allowed response
      res.status(405).json({ message: 'Method Not Allowed' });
    }
  }
  