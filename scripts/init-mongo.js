// MongoDB Initialization Script
// This runs on first container startup

print('==========================================');
print('VirtualFab MongoDB Initialization');
print('==========================================');

// Switch to virtualfab database
db = db.getSiblingDB('virtualfab');

// Create collections
db.createCollection('conversations');
db.createCollection('documents');
db.createCollection('users');
db.createCollection('analytics');

print('✅ Collections created');

// Create indexes
db.conversations.createIndex({ 'session_id': 1, 'timestamp': -1 });
db.conversations.createIndex({ 'user_id': 1 });
db.conversations.createIndex({ 'created_at': 1 }, { expireAfterSeconds: 2592000 }); // 30 days TTL

db.documents.createIndex({ 'domain': 1, 'uploaded_at': -1 });
db.documents.createIndex({ 'filename': 1 }, { unique: true });

db.users.createIndex({ 'email': 1 }, { unique: true });
db.users.createIndex({ 'username': 1 }, { unique: true });

db.analytics.createIndex({ 'event_type': 1, 'timestamp': -1 });
db.analytics.createIndex({ 'created_at': 1 }, { expireAfterSeconds: 7776000 }); // 90 days TTL

print('✅ Indexes created');

// Insert sample data
db.conversations.insertOne({
    session_id: 'sample-001',
    user_id: 'user-001',
    messages: [
        {
            role: 'user',
            content: '반도체 8대 공정에 대해 알려주세요',
            timestamp: new Date()
        },
        {
            role: 'assistant',
            content: '반도체 8대 공정은 다음과 같습니다...',
            timestamp: new Date()
        }
    ],
    created_at: new Date(),
    updated_at: new Date()
});

print('✅ Sample data inserted');

print('==========================================');
print('MongoDB initialization complete!');
print('==========================================');
